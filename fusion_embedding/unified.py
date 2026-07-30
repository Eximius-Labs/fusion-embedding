"""Canonical readout contract + a single ``UnifiedEmbedder`` interface (Phase A).

Before this module the "unified space" was three tiers with three readouts:

* Tier 1 (truly shared, frozen base + gated packs): text, image, video, audio, thermal.
* Tier 2 (projected in via a trained MLP): K3-vision, DINO geometry.
* Tier 3 (a DIFFERENT readout): Tremor motion.

They disagreed on output dim (1024 vs 2048), on whether whitening applied, and on the
instruction template, so vectors from different modalities were not even directly
comparable. This module fixes both defects in ONE place:

* :class:`ReadoutContract` is the single canonical read-out (dim policy, L2-norm,
  text-side whitening). Every ``embed_*`` finishes through ``contract.finish`` — there is
  no second place a modality's vector is shaped.
* :class:`UnifiedEmbedder` is the single interface. It loads the frozen base once,
  attaches the native paths (text/image/video), the audio resampler, the thermal pack,
  and the projector heads (motion / geometry / K3) as available, and manages the adapter
  gate so a forward never runs with the wrong pack open.

The contract, per :mod:`fusion_embedding.config`:

* interop default dim = full ``d_llm`` (2048 in production); the MRL rung (1024) is opt-in
  via ``dim=``. Uniform full-dim vectors make every modality directly comparable.
* L2-normalized (always).
* whitening is applied on the TEXT side only, before truncation (per-dim diagonal, so
  MRL-safe); every non-text modality is un-whitened. This matches how the model was
  trained (audio/image learn to match the whitened text target).
* per-modality instructions come from ``config.INSTRUCTION_REGISTRY``.
* :meth:`UnifiedEmbedder.center` (per-modality mean-centering) is the recommended default
  for cross-modal ranking.

The whole gate/contract core is exercised on CPU with the tiny stand-ins (see
``tests/test_unified.py``); the real weights are wired only in :meth:`from_pretrained`.
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Optional, Sequence, Union

import torch

from .config import (
    CHAT_TEMPLATE,
    IMAGE_USER_CONTENT,
    INSTRUCTION_REGISTRY,
    VIDEO_USER_CONTENT,
    FusionConfig,
)
from .model import last_token_pool, mrl_truncate_normalize

BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
AUDIO_MODEL = "Qwen/Qwen2.5-Omni-7B"

MODALITIES = ("text", "image", "video", "audio", "thermal", "motion", "geometry")


def _chat(instruction: str, content: str) -> str:
    """The base's official embedding format (single source: ``config.CHAT_TEMPLATE``)."""
    return CHAT_TEMPLATE.format(instruction=instruction, content=content)


# --------------------------------------------------------------------------- #
# The canonical read-out contract — the ONE place a vector is shaped.
# --------------------------------------------------------------------------- #
class ReadoutContract:
    """Single canonical read-out for every modality.

    ``finish`` is the only place output dim, whitening, and normalization are applied, so
    the contract cannot drift per modality. Interop default is full ``d_llm``; pass ``dim``
    to opt into a shorter MRL rung.
    """

    def __init__(self, dim: int, mrl_dims: Sequence[int], mrl_default: int):
        self.dim = int(dim)                       # interop default (full d_llm)
        self.mrl_dims = tuple(int(d) for d in mrl_dims)
        self.mrl_default = int(mrl_default)
        self.d_llm = max(self.mrl_dims)

    @classmethod
    def from_config(cls, cfg: FusionConfig) -> "ReadoutContract":
        # Interop default = full d_llm (== the top MRL rung) so every modality is one dim.
        return cls(dim=cfg.d_llm, mrl_dims=cfg.mrl_dims, mrl_default=cfg.mrl_default)

    def resolve_dim(self, dim: Optional[int]) -> int:
        d = int(dim) if dim else self.dim
        if d <= 0 or d > self.d_llm:
            raise ValueError(f"dim={d} must be in (0, d_llm={self.d_llm}]")
        return d

    def finish(self, pooled: torch.Tensor, *, is_text: bool = False,
               whitening: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
               dim: Optional[int] = None) -> torch.Tensor:
        """Raw pooled vector -> contract-compliant vector.

        Whitening (per-dim, diagonal) is applied on the text side only, on the RAW pooled
        vector, before truncation — MRL-safe because a diagonal map commutes with prefix
        truncation. Every other modality is left un-whitened. Always L2-normalized.
        """
        v = pooled.float()
        if is_text and whitening is not None:
            v = whitening(v)
        return mrl_truncate_normalize(v, self.resolve_dim(dim))


# --------------------------------------------------------------------------- #
# UnifiedEmbedder
# --------------------------------------------------------------------------- #
class UnifiedEmbedder:
    """One interface over every modality, on one frozen base, with one read-out.

    Construction has two modes that share ALL of the gate/contract logic:

    * DI / test mode (GPU-free): pass a :class:`~fusion_embedding.model.FusionEmbeddingModel`
      (real or tiny stand-in) plus the seams a modality needs — ``tokenizer`` for text, a
      ``vision_pooler`` for image/video/thermal, ``projectors`` for motion/geometry. Audio
      runs straight through the model's resampler + audio encoder.
    * Real mode: :meth:`from_pretrained` loads the frozen Qwen base and the released
      checkpoint, then attaches the real tokenizer/processor/feature-extractor, the thermal
      pack, and any projector heads.

    Gate discipline (the load-bearing invariant): text/image/video run with EVERY pack gate
    closed; audio opens only the audio gate; thermal opens only the thermal gate. Each
    ``embed_*`` opens its gate around its own forward and closes it after, so the mixed
    :meth:`embed` path never runs a forward with the wrong gate open.
    """

    def __init__(
        self,
        model,
        contract: Optional[ReadoutContract] = None,
        *,
        tokenizer=None,
        processor=None,
        audio_feature_extractor=None,
        base=None,                       # real Qwen3VLModel for the native vision path
        thermal_gate=None,               # an AdapterGate for the "thermal" pack (or None)
        projectors: Optional[dict] = None,   # name -> callable(input) -> feature[.., d_llm]
        vision_pooler: Optional[Callable] = None,  # DI seam: (kind, media) -> pooled [1,d_llm]
        device: str = "cpu",
    ):
        self.model = model
        self.cfg: FusionConfig = model.cfg
        self.contract = contract or ReadoutContract.from_config(self.cfg)
        self.tok = tokenizer
        self.proc = processor
        self.fe_audio = audio_feature_extractor
        self.full = base
        self.projectors = dict(projectors or {})
        self._vision_pooler = vision_pooler
        self.device = device
        self._audio_gate = getattr(model, "_adapter_gate", None)
        self._thermal_gate = thermal_gate

    # ----------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, repo_or_path: str, device: str = "cuda",
                        revision: Optional[str] = None,
                        thermal_repo: Optional[str] = None,
                        thermal_rank: int = 384,
                        projector_heads: Optional[dict] = None,
                        dtype=torch.bfloat16, **kw) -> "UnifiedEmbedder":
        """Load the frozen base + released checkpoint, then attach optional senses.

        ``thermal_repo`` attaches the Ember thermal pack via the multi-gate registry.
        ``projector_heads`` maps a name ("motion"/"geometry"/"k3") to a callable that takes
        the raw input and returns a base-space feature vector; heads are optional and lazily
        supplied so this stays a pure-search load by default. Reuses the released encoder
        code end to end (no encoder is reimplemented here)."""
        from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

        from .config import FusionConfig as _Cfg
        from .hf_components import BaseLMAdapter, load_audio_tower
        from .model import FusionEmbeddingModel
        import dataclasses

        # ---- checkpoint + config ----
        if os.path.exists(repo_or_path):
            path = repo_or_path
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
            path = None
            for name in ("fusion-embedding-2-2b-preview.pt", "fusion-embedding-1-2b-preview.pt"):
                try:
                    path = hf_hub_download(repo_or_path, name, revision=revision)
                    break
                except EntryNotFoundError:
                    continue
            if path is None:
                raise FileNotFoundError(f"no known checkpoint file in {repo_or_path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        flds = {f.name for f in dataclasses.fields(_Cfg)}
        cfg = _Cfg(**{k: v for k, v in ck["config"].items() if k in flds})

        full = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, dtype=dtype)
        full = full.to(device).eval()
        for p in full.parameters():
            p.requires_grad_(False)
        proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tower, _, _ = load_audio_tower(AUDIO_MODEL, device=device, dtype=dtype)
        fe_audio = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL, trust_remote_code=True)

        model = FusionEmbeddingModel(cfg, full.get_input_embeddings(),
                                     BaseLMAdapter(full.language_model), audio_encoder=tower)
        model.resampler.to(device).float()
        model.resampler.load_state_dict(ck["resampler"])
        if ("adapters" in ck) != (model.audio_adapters is not None):
            raise RuntimeError("adapter presence mismatch between checkpoint and config")
        if model.audio_adapters is not None:
            model.audio_adapters.to(device).float()
            model.audio_adapters.load_state_dict(ck["adapters"])
        model.text_whitening.load_state_dict(ck["text_whitening"])
        model.eval()

        thermal_gate = None
        if thermal_repo is not None:
            from .adapters import AdapterPacks
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            packs = AdapterPacks()
            adapters, thermal_gate = packs.add_pack("thermal", model.base_lm, cfg.d_llm, thermal_rank)
            src = thermal_repo if os.path.exists(thermal_repo) else hf_hub_download(thermal_repo, "model.safetensors")
            adapters.load_state_dict(load_file(src))
            packs.to(device)
            model._thermal_packs = packs  # keep a reference alive

        return cls(model, ReadoutContract.from_config(cfg), tokenizer=proc.tokenizer,
                   processor=proc, audio_feature_extractor=fe_audio, base=full,
                   thermal_gate=thermal_gate, projectors=projector_heads, device=device, **kw)

    # ------------------------------------------------------------ gate helpers
    def _any_gate_active(self) -> bool:
        a = self._audio_gate is not None and self._audio_gate.active
        t = self._thermal_gate is not None and self._thermal_gate.active
        return a or t

    def _require_all_closed(self, who: str) -> None:
        if self._any_gate_active():
            raise RuntimeError(f"{who} must run with every adapter gate closed; a gate is open")

    @contextlib.contextmanager
    def _only(self, gate):
        """Open ``gate`` (if any) for the block, asserting no OTHER pack gate is open."""
        others = [g for g in (self._audio_gate, self._thermal_gate) if g is not None and g is not gate]
        if any(g.active for g in others):
            raise RuntimeError("another adapter gate is already open")
        if gate is None:
            yield
        else:
            with gate:
                yield

    def _finish(self, pooled: torch.Tensor, *, is_text: bool = False,
                whitening=None, dim: Optional[int] = None) -> torch.Tensor:
        v = self.contract.finish(pooled, is_text=is_text, whitening=whitening, dim=dim)
        if v.dim() == 2 and v.size(0) == 1:
            v = v.squeeze(0)
        return v.cpu()

    # ------------------------------------------------------------------- text
    @torch.no_grad()
    def embed_text(self, text: str, instruction: Optional[str] = None,
                   dim: Optional[int] = None) -> torch.Tensor:
        """Embed text as a query. Whitened (text side), L2-normalized, contract dim."""
        self._require_all_closed("text embed")
        instr = instruction or INSTRUCTION_REGISTRY["text_query"]
        ids = self.tok.encode(_chat(instr, text), add_special_tokens=False)[:512]
        ids_t = torch.tensor([ids], device=self.device)
        pooled = self.model.encode_text(ids_t, torch.ones_like(ids_t))  # raises if a gate is open
        return self._finish(pooled, is_text=True, whitening=self.model.text_whitening, dim=dim)

    # ------------------------------------------------------ native vision path
    @torch.no_grad()
    def _vision_pooled(self, kind: str, media, *, instruction: str,
                       user_content: str, gate=None) -> torch.Tensor:
        """Run the frozen base's vision path -> pooled [1, d_llm], under ``gate`` if given.

        Non-target gates must be closed; that is asserted by :meth:`_only`. In DI/test mode a
        ``vision_pooler`` seam replaces the real processor+base (kept GPU-free)."""
        with self._only(gate):
            if self._vision_pooler is not None:
                return self._vision_pooler(kind, media)
            if self.full is None or self.proc is None:
                raise RuntimeError(f"{kind} embed needs a processor+base or a vision_pooler seam")
            from PIL import Image
            if isinstance(media, (str, os.PathLike)):
                media = Image.open(str(media))
            media = media.convert("RGB")           # single-channel thermal -> replicated to 3
            text = _chat(instruction, user_content)
            inputs = self.proc(text=[text], images=[media], return_tensors="pt").to(self.device)
            h = self.full(**inputs).last_hidden_state
            return last_token_pool(h, inputs["attention_mask"])

    @torch.no_grad()
    def embed_image(self, image, dim: Optional[int] = None) -> torch.Tensor:
        """Embed an image through the frozen base's native vision path (gates closed)."""
        pooled = self._vision_pooled("image", image,
                                     instruction=INSTRUCTION_REGISTRY["doc"],
                                     user_content=IMAGE_USER_CONTENT, gate=None)
        return self._finish(pooled, dim=dim)

    @torch.no_grad()
    def embed_video(self, video, fps: Optional[float] = None,
                    max_frames: Optional[int] = None, dim: Optional[int] = None) -> torch.Tensor:
        """Embed a video through the frozen base's native video path (gates closed).

        In real mode the released FE2 video preprocessing is reused via the injected
        ``vision_pooler``/processor; frame decoding is not reimplemented here."""
        if self._vision_pooler is not None:
            pooled = self._vision_pooled("video", video,
                                         instruction=INSTRUCTION_REGISTRY["doc"],
                                         user_content=VIDEO_USER_CONTENT, gate=None)
            return self._finish(pooled, dim=dim)
        if getattr(self, "_video_pooler", None) is None:
            raise NotImplementedError(
                "real video preprocessing is wired via from_pretrained's video seam; "
                "for the native path use fe2_release.inference.FusionEmbedder.embed_video")
        pooled = self._video_pooler(video, fps, max_frames)
        return self._finish(pooled, dim=dim)

    # ------------------------------------------------------------------ audio
    @torch.no_grad()
    def embed_audio(self, audio: Union[str, "torch.Tensor"], sr: Optional[int] = None,
                    dim: Optional[int] = None) -> torch.Tensor:
        """Embed audio (path or raw array). Opens ONLY the audio gate for the LM stage."""
        if self.fe_audio is None:
            raise RuntimeError("embed_audio needs a feature extractor; use embed_audio_mel for a mel")
        if isinstance(audio, (str, os.PathLike)):
            import soundfile as sf
            wav, sr = sf.read(str(audio), dtype="float32")
        else:
            import numpy as np
            wav = np.asarray(audio)
            assert sr is not None, "pass sr= when embedding a raw array"
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        target_sr = self.fe_audio.sampling_rate
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        feats = self.fe_audio(wav, sampling_rate=target_sr, return_tensors="pt",
                              return_attention_mask=True, padding="max_length", truncation=True)
        mel = feats["input_features"][0]
        am = feats.get("attention_mask")
        if am is not None:
            mel = mel[:, : int(am[0].sum().item())]
        return self.embed_audio_mel(mel, dim=dim)

    @torch.no_grad()
    def embed_audio_mel(self, mel: torch.Tensor, mel_mask: Optional[torch.Tensor] = None,
                        dim: Optional[int] = None) -> torch.Tensor:
        """Low-level audio seam: a mel window [n_mels, F] (or [1, n_mels, F]) -> vector.

        This is the tested core: it resamples the mel to N latent tokens and runs the frozen
        LM with ONLY the audio gate open (via ``model.encode_audio``)."""
        if self._thermal_gate is not None and self._thermal_gate.active:
            raise RuntimeError("thermal gate is open during an audio embed")
        mel = mel.to(self.device)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        B, _, F = mel.shape
        if mel_mask is None:
            mel_mask = torch.ones(B, F, dtype=torch.bool, device=self.device)
        audio_tok = self.model.audio_tokens(mel, mel_mask)          # [B, N, d_llm]
        ids = torch.tensor([[self.cfg.audio_pad_id] * self.cfg.n_query + [self.cfg.eos_id]],
                           device=self.device).expand(B, -1)
        pooled = self.model.encode_audio(ids, torch.ones_like(ids), audio_tok)  # opens audio gate
        return self._finish(pooled, dim=dim)

    # ---------------------------------------------------------------- thermal
    @torch.no_grad()
    def embed_thermal(self, image, dim: Optional[int] = None) -> torch.Tensor:
        """Embed a thermal image via the vision path with ONLY the thermal pack open."""
        if self._thermal_gate is None:
            raise RuntimeError("thermal pack not loaded (pass thermal_repo= / thermal_gate=)")
        pooled = self._vision_pooled("thermal", image,
                                     instruction=INSTRUCTION_REGISTRY["thermal"],
                                     user_content=IMAGE_USER_CONTENT, gate=self._thermal_gate)
        return self._finish(pooled, dim=dim)

    # ------------------------------------------------------ projector heads
    @torch.no_grad()
    def _project(self, name: str, x, dim: Optional[int] = None) -> torch.Tensor:
        head = self.projectors.get(name)
        if head is None:
            raise RuntimeError(f"projector head {name!r} not loaded; available: {list(self.projectors)}")
        feat = head(x)                          # base-space feature [.., d_llm]
        if not torch.is_tensor(feat):
            feat = torch.as_tensor(feat)
        return self._finish(feat, dim=dim)

    @torch.no_grad()
    def embed_motion(self, accel, dim: Optional[int] = None) -> torch.Tensor:
        """Embed a motion window (Tremor head). NOTE: the motion projector was fit against a
        DIFFERENT text template (config.MOTION_TEXT_TEMPLATE); rank motion against text with
        the Tremor text readout, not this interface's ``embed_text`` (see the contract doc)."""
        return self._project("motion", accel, dim=dim)

    @torch.no_grad()
    def embed_geometry(self, image, dim: Optional[int] = None) -> torch.Tensor:
        """Embed an image's DINO geometry (Fusion Perception head) into the shared space."""
        return self._project("geometry", image, dim=dim)

    # ----------------------------------------------------- mixed / grouped API
    @torch.no_grad()
    def embed(self, items: Sequence, dim: Optional[int] = None) -> list:
        """Embed a mixed list, GROUPED BY MODALITY so each pack gate opens at most once per
        group. ``items`` are dicts ``{"modality", "input", ...}`` or ``(modality, input)``
        tuples. Returns vectors in the original order."""
        parsed = [self._parse_item(it) for it in items]
        order: list = []
        groups: dict = {}
        for i, (m, payload, kw) in enumerate(parsed):
            groups.setdefault(m, []).append((i, payload, kw))
            if m not in order:
                order.append(m)
        out: list = [None] * len(items)
        for m in order:                                   # one modality fully, then the next
            fn = self._dispatch(m)
            for i, payload, kw in groups[m]:
                out[i] = fn(payload, dim=dim, **kw)
        return out

    def _parse_item(self, it):
        if isinstance(it, dict):
            kw = {k: v for k, v in it.items() if k not in ("modality", "input")}
            return it["modality"], it["input"], kw
        m, payload = it
        return m, payload, {}

    def _dispatch(self, modality: str) -> Callable:
        table = {
            "text": self.embed_text, "image": self.embed_image, "video": self.embed_video,
            "audio": self.embed_audio, "thermal": self.embed_thermal,
            "motion": self.embed_motion, "geometry": self.embed_geometry,
        }
        if modality not in table:
            raise ValueError(f"unknown modality {modality!r}; known: {tuple(table)}")
        return table[modality]

    # ----------------------------------------------------- cross-modal readout
    @staticmethod
    def center(embs: torch.Tensor) -> torch.Tensor:
        """Per-modality mean-centering + renormalization. The recommended default when
        ranking a gallery of one modality against queries of another (a few points of R@1
        across modality pairs in evaluation)."""
        c = embs - embs.mean(dim=0, keepdim=True)
        return torch.nn.functional.normalize(c, dim=-1)

    @staticmethod
    def rank_cross_modal(queries: torch.Tensor, gallery: torch.Tensor,
                         center: bool = True) -> torch.Tensor:
        """Cosine score matrix [n_query, n_gallery], mean-centering each side by default."""
        q = UnifiedEmbedder.center(queries) if center else queries
        g = UnifiedEmbedder.center(gallery) if center else gallery
        return q @ g.T
