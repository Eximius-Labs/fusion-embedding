"""UnifiedEmbedder + ReadoutContract — GPU-free, on the tiny CPU stand-ins.

Verifies the two Phase-A guarantees on the single interface:

(a) every modality emits a contract-compliant vector (shape / dtype / L2-norm), and the
    MRL opt-in shortens the dim;
(b) with a foreign gate closed, text/image forwards are BITWISE identical to the frozen
    base (``torch.equal``), and cycling the audio / thermal pack gates does not perturb the
    other modalities;
(c) the mixed-modality path groups by modality and never runs a forward with the wrong
    gate open (checked by a decoder-layer hook that records the live gate state).

No GPU / transformers: text uses a trivial tokenizer stand-in, image/video/thermal use an
injected vision-pooler seam over the tiny base, audio uses the tiny mel path, and the
projector heads (motion / geometry) are injected callables.
"""

import torch
import torch.nn as nn
import pytest

from fusion_embedding.config import FusionConfig
from fusion_embedding._tiny import build_tiny_model
from fusion_embedding.adapters import AdapterPacks, find_decoder_layers
from fusion_embedding.model import last_token_pool, mrl_truncate_normalize
from fusion_embedding.unified import ReadoutContract, UnifiedEmbedder

RANK = 8


# --------------------------------------------------------------------------- #
# tiny stand-in seams
# --------------------------------------------------------------------------- #
class TinyTokenizer:
    """Deterministic char-hash tokenizer producing ids in [3, vocab)."""

    def __init__(self, vocab: int = 64):
        self.vocab = vocab

    def encode(self, s: str, add_special_tokens: bool = False):
        return [3 + (ord(c) % (self.vocab - 3)) for c in s][:64] or [3]


class TinyFeatureExtractor:
    """A stand-in for the Whisper feature extractor: waveform -> a fixed tiny mel."""

    def __init__(self, cfg, sampling_rate: int = 16000, n_frames: int = 20):
        self.sampling_rate = sampling_rate
        self.n_mels = cfg.n_mels
        self.n_frames = n_frames

    def __call__(self, wav, sampling_rate, return_tensors="pt",
                 return_attention_mask=True, padding=None, truncation=None):
        mel = torch.linspace(-0.5, 0.5, self.n_mels * self.n_frames).reshape(self.n_mels, self.n_frames)
        return {"input_features": mel.unsqueeze(0),
                "attention_mask": torch.ones(1, self.n_frames, dtype=torch.long)}


def _randomize_up(module: nn.Module, seed: int):
    """Give the zero-inited 'up' projections nonzero weights (an engaged pack)."""
    g = torch.Generator().manual_seed(seed)
    for m in module.modules():
        if isinstance(m, nn.Linear) and torch.count_nonzero(m.weight) == 0 \
                and m.weight.shape[0] >= m.weight.shape[1]:
            m.weight.data = torch.randn(m.weight.shape, generator=g) * 0.02


def _make_vision_pooler(model):
    """A GPU-free stand-in for the native vision path: token ids -> pooled base hidden.

    Runs the (hook-carrying) tiny base, so the surrounding gate scope decides whether the
    thermal adapters fire — exactly as the real vision forward runs under packs.scope."""
    def pooler(kind, media):
        ids = media if media.dim() == 2 else media.unsqueeze(0)
        mask = torch.ones_like(ids)
        embeds = model.embed_tokens(ids)
        hidden = model.base_lm(inputs_embeds=embeds, attention_mask=mask)
        return last_token_pool(hidden, mask)
    return pooler


def _fake_media(cfg, seed=5, S=6):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 60, (S,), generator=g)
    return ids


def _fake_mel(cfg, seed=7, Fdim=20):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(cfg.n_mels, Fdim, generator=g)


def build_unified(adapter_rank=RANK, with_thermal=True, seed=0):
    """A tiny UnifiedEmbedder with an engaged audio pack and (optionally) a thermal pack."""
    cfg = FusionConfig.tiny(adapter_rank=adapter_rank)
    model = build_tiny_model(cfg, seed=seed)
    if model.audio_adapters is not None:
        _randomize_up(model.audio_adapters, seed=101)   # engaged (nonzero) audio pack
    thermal_gate = None
    if with_thermal:
        packs = AdapterPacks()
        adapters, thermal_gate = packs.add_pack("thermal", model.base_lm, cfg.d_llm, RANK)
        _randomize_up(adapters, seed=202)               # engaged thermal pack
        model._thermal_packs = packs
    projectors = {
        "motion": lambda x: torch.full((cfg.d_llm,), 0.5) + torch.arange(cfg.d_llm).float() * 0.01,
        "geometry": lambda x: torch.linspace(-1, 1, cfg.d_llm),
    }
    ue = UnifiedEmbedder(
        model, ReadoutContract.from_config(cfg),
        tokenizer=TinyTokenizer(), audio_feature_extractor=TinyFeatureExtractor(cfg),
        thermal_gate=thermal_gate,
        projectors=projectors, vision_pooler=_make_vision_pooler(model), device="cpu",
    )
    return cfg, ue


def _ref_finish_tokens(ref_model, ids_1d, is_text, dim=None):
    ids = ids_1d.unsqueeze(0)
    mask = torch.ones_like(ids)
    embeds = ref_model.embed_tokens(ids)
    hidden = ref_model.base_lm(inputs_embeds=embeds, attention_mask=mask)
    pooled = last_token_pool(hidden, mask)
    contract = ReadoutContract.from_config(ref_model.cfg)
    wh = ref_model.text_whitening if is_text else None
    return contract.finish(pooled, is_text=is_text, whitening=wh, dim=dim).squeeze(0)


# --------------------------------------------------------------------------- #
# (0) the contract itself
# --------------------------------------------------------------------------- #
def test_contract_defaults_full_dim_and_norm():
    cfg = FusionConfig.tiny()
    c = ReadoutContract.from_config(cfg)
    assert c.dim == cfg.d_llm                       # interop default = full d_llm
    v = c.finish(torch.randn(1, cfg.d_llm))
    assert v.shape == (1, cfg.d_llm)
    assert torch.allclose(v.norm(), torch.tensor(1.0), atol=1e-5)


def test_contract_mrl_opt_in_and_text_only_whitening():
    cfg = FusionConfig.tiny()
    c = ReadoutContract.from_config(cfg)
    rung = 16
    assert rung in cfg.mrl_dims
    assert c.finish(torch.randn(1, cfg.d_llm), dim=rung).shape == (1, rung)

    # a whitening that would visibly change the vector, applied ONLY on the text side
    shift = lambda x: x + 10.0
    pooled = torch.randn(1, cfg.d_llm)
    doc = c.finish(pooled, is_text=False, whitening=shift)
    txt = c.finish(pooled, is_text=True, whitening=shift)
    assert not torch.equal(doc, txt)                        # whitening changed text
    assert torch.equal(doc, c.finish(pooled, is_text=False))  # ignored for non-text


# --------------------------------------------------------------------------- #
# (a) every modality is contract-compliant
# --------------------------------------------------------------------------- #
def test_every_modality_emits_contract_shape():
    cfg, ue = build_unified()
    d = cfg.d_llm
    media = _fake_media(cfg)
    outs = {
        "text": ue.embed_text("a dog barking on a beach"),
        "image": ue.embed_image(media),
        "audio": ue.embed_audio_mel(_fake_mel(cfg)),
        "thermal": ue.embed_thermal(media),
        "motion": ue.embed_motion(torch.randn(3, 50)),
        "geometry": ue.embed_geometry(media),
    }
    for name, v in outs.items():
        assert v.shape == (d,), f"{name}: {tuple(v.shape)} != ({d},)"
        assert v.dtype == torch.float32, f"{name} dtype {v.dtype}"
        assert torch.allclose(v.norm(), torch.tensor(1.0), atol=1e-5), f"{name} not unit-norm"


def test_mrl_opt_in_shortens_every_modality():
    cfg, ue = build_unified()
    rung = 16
    for v in (ue.embed_text("x", dim=rung), ue.embed_image(_fake_media(cfg), dim=rung),
              ue.embed_audio_mel(_fake_mel(cfg), dim=rung),
              ue.embed_motion(torch.randn(3, 40), dim=rung)):
        assert v.shape == (rung,)


# --------------------------------------------------------------------------- #
# (b) bitwise invariance of non-target forwards
# --------------------------------------------------------------------------- #
def test_closed_gate_forwards_are_bitwise_base():
    """text + image (all gates closed, engaged packs present) == the pack-free base."""
    cfg, ue = build_unified()
    ref = build_tiny_model(FusionConfig.tiny(), seed=0)      # same seed -> same base weights
    media = _fake_media(cfg)

    txt = ue.embed_text("hello world")
    ids = ue.tok.encode(
        "<|im_start|>system\nRetrieve images or text relevant to the user's query.<|im_end|>\n"
        "<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n")
    ref_txt = _ref_finish_tokens(ref, torch.tensor(ids[:512]), is_text=True)
    assert torch.equal(txt, ref_txt), "text forward not bitwise-identical to the frozen base"

    img = ue.embed_image(media)
    ref_img = _ref_finish_tokens(ref, media, is_text=False)
    assert torch.equal(img, ref_img), "image forward not bitwise-identical to the frozen base"


def test_gate_cycles_do_not_perturb_other_modalities():
    cfg, ue = build_unified()
    media = _fake_media(cfg)
    before_txt = ue.embed_text("hello world")
    before_img = ue.embed_image(media)

    # cycle the audio and thermal gates via their own encodes
    ue.embed_audio_mel(_fake_mel(cfg))
    ue.embed_thermal(media)
    assert not ue._any_gate_active(), "a gate was left open after an encode"

    assert torch.equal(before_txt, ue.embed_text("hello world"))
    assert torch.equal(before_img, ue.embed_image(media))


def test_thermal_pack_actually_engages():
    """Same image bytes: thermal (thermal gate open) must differ from image (gates closed)."""
    cfg, ue = build_unified()
    media = _fake_media(cfg)
    assert not torch.equal(ue.embed_image(media), ue.embed_thermal(media))


def test_text_refuses_when_a_foreign_gate_is_open():
    cfg, ue = build_unified()
    ue._thermal_gate.__enter__()                    # force the thermal gate open
    try:
        with pytest.raises(RuntimeError):
            ue.embed_text("should refuse")
        with pytest.raises(RuntimeError):
            ue.embed_image(_fake_media(cfg))
    finally:
        ue._thermal_gate.__exit__()


# --------------------------------------------------------------------------- #
# (c) mixed path groups by modality with correct gate state
# --------------------------------------------------------------------------- #
def test_mixed_path_groups_and_gates_are_exclusive():
    cfg, ue = build_unified()
    media = _fake_media(cfg)

    records = []   # (audio_active, thermal_active) at each decoder-layer forward
    layer0 = find_decoder_layers(ue.model.base_lm)[0]
    handle = layer0.register_forward_hook(
        lambda *_: records.append(
            (ue._audio_gate is not None and ue._audio_gate.active,
             ue._thermal_gate is not None and ue._thermal_gate.active)))

    wav = torch.linspace(-1, 1, 4000).numpy()   # raw waveform at the stub's sample rate
    items = [
        ("text", "a dog"),
        ("image", media),
        {"modality": "audio", "input": wav, "sr": 16000},
        ("thermal", media),
        ("motion", torch.randn(3, 30)),
        ("geometry", media),
        ("text", "a cat"),
    ]
    try:
        out = ue.embed(items)
    finally:
        handle.remove()

    assert len(out) == len(items)
    assert all(v.shape == (cfg.d_llm,) for v in out)
    assert not ue._any_gate_active()

    # never both gates open at once
    assert all(not (a and t) for a, t in records)
    # base forwards: text x2, image x1, audio x1, thermal x1 (motion/geometry are projector
    # heads and never touch the base). Exactly one has the audio gate open, one the thermal.
    assert sum(1 for a, t in records if a) == 1, records
    assert sum(1 for a, t in records if t) == 1, records
    assert sum(1 for a, t in records if not a and not t) == 3, records
