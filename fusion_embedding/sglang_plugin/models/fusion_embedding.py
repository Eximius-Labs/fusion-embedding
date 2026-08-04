"""FusionEmbeddingModel — SGLang pooling implementation of fusion-embedding-2.

Architecture recap (mirrors the released ``modeling_fusion_embedding.py``):

* FROZEN base: Qwen/Qwen3-VL-Embedding-2B (weights come from the base repo — the
  fusion repo carries only the trained parts). Text/image/video run bit-for-bit the
  frozen base path: last-token pooling, then (text only) diagonal whitening, then
  MRL truncation + L2 normalization.
* Trained parts (the fusion repo's ``model.safetensors``): a perceiver-resampler
  connector (mel-frame features -> 64 latent tokens in the LLM space), a diagonal
  text whitening, and modality-gated bottleneck adapters (rank 384) on all 28
  decoder layers that execute ONLY on tokens of audio sequences.
* FROZEN audio tower: the Qwen2.5-Omni-7B ``thinker.audio_tower`` encoder
  (d_audio 3584, post-projection features), also loaded from its own repo.

Audio is served as an SGLang multimodal input: the prompt carries ONE
``<|vision_pad|>`` placeholder per clip (token id 151654 — the base tokenizer has no
``<|audio_pad|>`` string; the released checkpoint reuses this inert token as the
audio slot marker). The processor expands it to ``n_query`` (64) slots and attaches
the 30 s Whisper mel; ``get_audio_feature`` runs the audio tower + resampler to
produce the 64 injected embeddings, and the client appends ``<|im_end|>`` so the
pooled position matches the reference readout (``[audio_pad] * 64 + [eos]``).

Positions: SGLang computes M-RoPE positions from the token ids at processing time
(``get_rope_index``); the audio slots carry no vision markers, so they get plain
sequential positions in all three rope rows — exactly the reference behavior. The
deepstack buffer is zero-initialized and only image/video embeddings write into it,
so audio positions receive the same zero multiscale channels as any text position
(both handled by SGLang itself; unlike the vLLM port, no explicit feature-drop or
zero-padding is needed here).

Gating in a continuously-batched engine cannot be a global on/off switch, so the
adapter hooks are gated PER TOKEN: ``forward`` marks the flattened token rows that
belong to requests carrying an audio item (the served audio prompt is exactly the 64
slots plus the trailing ``<|im_end|>``) and the decoder-layer hooks touch only those
rows. Rows of non-audio requests are never read or written, preserving the family's
byte-identity guarantee for text/image/video. The plugin's ServerArgs hook forces
eager execution so these Python hooks can never be baked into a captured graph.
"""

from __future__ import annotations

import glob
import logging
import math
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_AUDIO_MODEL = "Qwen/Qwen2.5-Omni-7B"

# The base tokenizer string whose id (151654) the released checkpoint reuses as the
# audio placeholder ("<|audio_pad|>" does not exist in the base vocabulary).
AUDIO_PLACEHOLDER = "<|vision_pad|>"

_ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


def _audio_enabled() -> bool:
    return os.environ.get("FUSION_SGLANG_DISABLE_AUDIO", "0") != "1"


def _adapters_enabled() -> bool:
    """Stage-gating escape hatch for the smoke test: FUSION_SGLANG_DISABLE_ADAPTERS=1
    skips the gated adapters entirely (frozen-base-only, stage A). The audio path
    REQUIRES the adapters (the released checkpoint was trained with them), so audio
    refuses to run with adapters disabled."""
    return os.environ.get("FUSION_SGLANG_DISABLE_ADAPTERS", "0") != "1"


# --------------------------------------------------------------------------- #
# Trained modules — verbatim mirrors of the released modeling file.
# --------------------------------------------------------------------------- #
def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 2 != 0:
        pe = sinusoidal_positions(length, dim + 1, device, dtype)
        return pe[:, :dim]
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class TextWhitening(nn.Module):
    """Diagonal (per-dim, MRL-safe) standardization of frozen text embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.uint8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(self.fitted) == 0:
            return x
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std


class _ResamplerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, q, kv, key_padding_mask):
        h = self.norm_sa(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        q = q + self.cross_attn(h, kv_n, kv_n, key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        q = q + self.ffn(self.norm_ff(q))
        return q


class FusionResampler(nn.Module):
    """Perceiver-resampler: variable-length audio frames -> N fixed latent tokens."""

    def __init__(self, d_audio: int, d_resampler: int, n_query: int, depth: int,
                 heads: int, ffn_mult: int, dropout: float, d_llm: int):
        super().__init__()
        self.in_proj = nn.Linear(d_audio, d_resampler)
        self.queries = nn.Parameter(torch.empty(n_query, d_resampler))
        nn.init.normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList(
            _ResamplerBlock(d_resampler, heads, ffn_mult, dropout) for _ in range(depth)
        )
        self.out_proj = nn.Linear(d_resampler, d_llm)
        self.out_norm = nn.LayerNorm(d_llm)

    def forward(self, frames: torch.Tensor, frame_mask: Optional[torch.Tensor] = None):
        B, T, _ = frames.shape
        if frame_mask is None:
            frame_mask = torch.ones(B, T, dtype=torch.bool, device=frames.device)
        kv = self.in_proj(frames)
        kv = kv + sinusoidal_positions(T, kv.size(-1), kv.device, kv.dtype).unsqueeze(0)
        key_padding = ~frame_mask
        fully_masked = key_padding.all(dim=1)
        if fully_masked.any():
            key_padding = key_padding.clone()
            key_padding[fully_masked, 0] = False
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            q = block(q, kv, key_padding)
        return self.out_norm(self.out_proj(q))


class GatedAdapter(nn.Module):
    """Parallel bottleneck adapter ``h + up(act(down(LN(h))))``, computed in fp32."""

    def __init__(self, d_model: int, rank: int, act: str = "silu"):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank, bias=False)
        self.act = _ACTS[act]()
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(h.float())))).to(h.dtype)


# --------------------------------------------------------------------------- #
# Token-gated adapter hooks on SGLang decoder layers
# --------------------------------------------------------------------------- #
class _AdapterState:
    """Per-forward flattened-row indices of tokens belonging to audio requests."""

    __slots__ = ("idx",)

    def __init__(self) -> None:
        self.idx: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.idx = None


def _make_layer_hook(adapter: GatedAdapter, state: _AdapterState):
    """Forward hook for an SGLang Qwen3 decoder layer.

    SGLang layers return ``(hidden_states, residual)`` — the residual add is fused
    into the NEXT layer's input layernorm (or the final norm), so the full residual
    stream after layer i is ``hidden_states + residual``. The reference applies
    ``h + adapter(h)`` to the full stream, so here: ``full = hidden + residual;
    hidden' = hidden + adapter(full)`` which yields ``full' = full + adapter(full)``
    downstream. Only rows in ``state.idx`` (audio requests) are read or written; with
    no audio rows the hook is a pure no-op, so non-audio forwards remain bitwise
    those of the frozen base.
    """

    def hook(_module, _inputs, output):
        idx = state.idx
        if idx is None or idx.numel() == 0:
            return None
        hidden, residual = output
        n = hidden.shape[0]
        use = idx[idx < n]
        if use.numel() == 0:
            return None
        sub = hidden.index_select(0, use)
        if residual is None:
            full = sub
        else:
            full = sub + residual.index_select(0, use)
        hidden.index_copy_(0, use, sub + adapter(full))
        return None

    return hook


# --------------------------------------------------------------------------- #
# Pooler: last-token pool -> (text only) whitening -> MRL truncate -> L2 norm
# --------------------------------------------------------------------------- #
class FusionPooler(nn.Module):
    """Modality-aware replacement for SGLang's plain LAST pooler.

    Reproduces the released readout per modality. The base tokenizer appends
    ``<|endoftext|>`` (the pad id) when called with the default
    ``add_special_tokens=True``. The reference readout tokenizes TEXT with
    ``add_special_tokens=False`` (pools at the assistant-opener newline) but
    images/videos through the HF processor default (pools AT the appended
    ``<|endoftext|>``); reference audio has no trailing pad (pools at ``<|im_end|>``).
    Attention is causal, so the hidden state one position before a trailing pad is
    bit-identical to a no-pad prompt's last hidden — therefore: step back over
    exactly one trailing pad for sequences WITHOUT vision items (text and audio);
    keep the true last token for vision sequences.

    Note: modality is decided from ``forward_batch.mm_inputs`` (not token ids —
    SGLang replaces multimodal token positions with per-item hash values for prefix
    matching, so the original placeholder ids are not visible here). It must be
    SNAPSHOT before the forward runs: ``general_mm_embed_routine`` sets
    ``forward_batch.mm_inputs = None`` once the embeddings are injected, so by
    pooling time the batch looks all-text. ``FusionEmbeddingModel.forward`` calls
    :meth:`snapshot_batch_modalities` up front.
    """

    def __init__(self, whitening: TextWhitening, trailing_pad_id: int):
        super().__init__()
        self.whitening = whitening
        self.trailing_pad_id = int(trailing_pad_id)
        # (has_vision, has_mm) per request of the in-flight forward.
        self._modalities: Optional[List[Tuple[bool, bool]]] = None

    def snapshot_batch_modalities(self, forward_batch: ForwardBatch) -> None:
        mm_list = forward_batch.mm_inputs
        seq_lens = forward_batch.extend_seq_lens_cpu
        if mm_list is None or seq_lens is None:
            self._modalities = None
            return
        flags: List[Tuple[bool, bool]] = []
        for i in range(len(seq_lens)):
            mm = mm_list[i] if i < len(mm_list) else None
            items = list(mm.mm_items) if mm is not None else []
            flags.append((any(it.is_image() or it.is_video() for it in items),
                          len(items) > 0))
        self._modalities = flags

    def clear_batch_modalities(self) -> None:
        self._modalities = None

    def forward(self, hidden_states: torch.Tensor,
                forward_batch: ForwardBatch) -> EmbeddingPoolerOutput:
        seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        last_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        n_reqs = len(seq_lens_cpu)
        modalities = self._modalities
        if modalities is None or len(modalities) != n_reqs:
            modalities = [(False, False)] * n_reqs

        last_ids = forward_batch.input_ids[last_indices].tolist()
        offsets: List[int] = []
        text_flags: List[bool] = []
        for i in range(n_reqs):
            has_vision, has_mm = modalities[i]
            step_back = (
                not has_vision
                and int(seq_lens_cpu[i]) >= 2   # never step across a chunk boundary
                and int(last_ids[i]) == self.trailing_pad_id
            )
            offsets.append(1 if step_back else 0)
            text_flags.append(not has_mm)
        if any(offsets):
            last_indices = last_indices - torch.tensor(
                offsets, device=last_indices.device, dtype=last_indices.dtype
            )

        pooled = hidden_states[last_indices].to(torch.float32)

        if any(text_flags):
            whitened = self.whitening(pooled)
            sel = torch.tensor(text_flags, device=pooled.device).unsqueeze(-1)
            pooled = torch.where(sel, whitened, pooled)

        # Matryoshka truncation, then L2 normalization — the same order as the
        # reference contract (whiten raw, truncate, normalize).
        dims = forward_batch.dimensions
        if dims is not None and len(set(dims)) == 1:
            pooled = pooled[..., : dims[0]]
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        elif dims is not None:
            pooled = [
                nn.functional.normalize(row[:dim], p=2, dim=-1)
                for row, dim in zip(pooled, dims)
            ]
        else:
            pooled = nn.functional.normalize(pooled, p=2, dim=-1)
        return EmbeddingPoolerOutput(embeddings=pooled)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class FusionEmbeddingModel(Qwen3VLForConditionalGeneration):
    # The fusion repo also carries the reference ``*.pt`` checkpoint; never let the
    # loader fall back to it (the safetensors file has everything trained).
    fall_back_to_pt_during_load = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        cfg = config
        serving_dtype = torch.get_default_dtype()

        self.fusion_base_model = getattr(cfg, "base_model", DEFAULT_BASE_MODEL)
        self.fusion_audio_model = getattr(cfg, "audio_model", DEFAULT_AUDIO_MODEL)
        self.fusion_audio_pad_id = int(getattr(cfg, "audio_pad_id", 151654))
        self.fusion_eos_id = int(getattr(cfg, "eos_id", 151645))
        d_llm = int(getattr(cfg, "d_llm", cfg.text_config.hidden_size))

        # ---- trained modules (fp32, like the reference runtime) ----
        self.resampler = FusionResampler(
            d_audio=int(getattr(cfg, "d_audio", 3584)),
            d_resampler=int(getattr(cfg, "d_resampler", 384)),
            n_query=int(getattr(cfg, "n_query", 64)),
            depth=int(getattr(cfg, "resampler_depth", 6)),
            heads=int(getattr(cfg, "resampler_heads", 8)),
            ffn_mult=int(getattr(cfg, "resampler_ffn_mult", 4)),
            dropout=float(getattr(cfg, "resampler_dropout", 0.0)),
            d_llm=d_llm,
        ).float()
        self.resampler.eval()
        self.text_whitening = TextWhitening(d_llm).float()

        # ---- gated adapters on the decoder layers ----
        self.audio_adapters: Optional[nn.ModuleList] = None
        self._adapter_state = _AdapterState()
        adapter_rank = int(getattr(cfg, "adapter_rank", 0) or 0)
        if adapter_rank > 0 and not _adapters_enabled():
            logger.warning("[fusion] gated adapters DISABLED via env — frozen-base "
                           "only; the audio path is unavailable in this configuration")
            adapter_rank = 0
        if adapter_rank > 0:
            layers = self.model.layers
            n_expected = int(getattr(cfg, "n_decoder_layers", len(layers)))
            if len(layers) != n_expected:
                raise RuntimeError(
                    f"decoder has {len(layers)} layers but the checkpoint expects "
                    f"{n_expected} adapters"
                )
            adapter_act = getattr(cfg, "adapter_act", "silu")
            self.audio_adapters = nn.ModuleList(
                GatedAdapter(d_llm, adapter_rank, adapter_act) for _ in layers
            ).float()
            self.audio_adapters.eval()
            self._adapter_handles = [
                layer.register_forward_hook(_make_layer_hook(ad, self._adapter_state))
                for layer, ad in zip(layers, self.audio_adapters)
            ]

        # ---- frozen audio tower (Qwen2.5-Omni thinker.audio_tower) ----
        self.audio_tower = None
        if _audio_enabled():
            from transformers import AutoConfig as HFAutoConfig
            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                Qwen2_5OmniAudioEncoder,
            )

            acfg = HFAutoConfig.from_pretrained(self.fusion_audio_model)
            audio_cfg = acfg.thinker_config.audio_config
            # Follow the serving dtype (bf16 by default), like the reference's
            # load_audio_tower(dtype=...).
            self.audio_tower = Qwen2_5OmniAudioEncoder(audio_cfg).to(serving_dtype)
            self.audio_tower.eval()
            if int(getattr(cfg, "adapter_rank", 0) or 0) > 0 and self.audio_adapters is None:
                raise RuntimeError(
                    "the checkpoint's audio path requires its gated adapters; unset "
                    "FUSION_SGLANG_DISABLE_ADAPTERS or set FUSION_SGLANG_DISABLE_AUDIO=1"
                )

        # ---- pooler: replace the stock LAST pooler with the readout contract ----
        self.pooler = FusionPooler(
            self.text_whitening,
            trailing_pad_id=int(getattr(cfg, "pad_id", 151643)),
        )

    # ------------------------------------------------------------- audio path
    @torch.no_grad()
    def _omni_frames(self, mel: torch.Tensor) -> torch.Tensor:
        """One clip's trimmed mel [n_mels, L] -> frames [T, d_audio] (fp32).

        Mirrors the reference OmniAudioAdapter (B=1, post_proj features).
        ``aftercnn_lens`` is passed explicitly: required by some transformers
        versions and ignored by others; the explicit value equals the internal
        computation ((L - 1) // 2 + 1 accumulated over the encoder's chunking).
        """
        assert self.audio_tower is not None, \
            "audio path disabled (FUSION_SGLANG_DISABLE_AUDIO)"
        tower_dtype = next(self.audio_tower.parameters()).dtype
        L = int(mel.shape[-1])
        n2 = 2 * int(self.audio_tower.n_window)
        n_full, tail = divmod(L, n2)
        chunk_lens = [n2] * n_full + ([tail] if tail else [])
        aftercnn = sum((c - 1) // 2 + 1 for c in chunk_lens)
        out = self.audio_tower(
            input_features=mel.to(tower_dtype),
            feature_lens=torch.tensor([L], device=mel.device),
            aftercnn_lens=torch.tensor([aftercnn], device=mel.device),
        )
        frames = out.last_hidden_state
        if frames.dim() == 3:
            frames = frames[0]
        return frames.float()

    @torch.no_grad()
    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """SGLang multimodal hook: audio items -> [n_items * n_query, hidden] tokens.

        The deepstack multiscale buffer is zero-initialized by SGLang and audio is
        not registered in ``use_deepstack``, so audio positions receive the same
        zero multiscale channels as text positions — no explicit padding needed
        (this is where the vLLM port had to zero-pad the embedding channels)."""
        if self.audio_adapters is None:
            raise RuntimeError(
                "the audio path requires the checkpoint's gated adapters "
                "(FUSION_SGLANG_DISABLE_ADAPTERS is set?)"
            )
        device = next(self.model.parameters()).device
        target_dtype = self.model.embed_tokens.weight.dtype
        out: List[torch.Tensor] = []
        for item in items:
            mel = item.feature
            if not isinstance(mel, torch.Tensor):
                mel = torch.as_tensor(mel)
            mel = mel.to(device=device, dtype=torch.float32)
            lens = item.model_specific_data.get("audio_feature_lens")
            if lens is None:
                lens_list = [int(mel.shape[-1])] * (mel.shape[0] if mel.dim() == 3 else 1)
            elif isinstance(lens, torch.Tensor):
                lens_list = [int(x) for x in lens.reshape(-1).tolist()]
            else:
                lens_list = [int(x) for x in lens]
            clips = list(mel.unbind(0)) if mel.dim() == 3 else [mel]
            if len(clips) != len(lens_list):
                raise RuntimeError(
                    f"audio item carries {len(clips)} mel(s) but "
                    f"{len(lens_list)} length(s); serve one clip per request"
                )
            for clip, L in zip(clips, lens_list):
                clip = clip[..., : max(L, 1)]
                frames = self._omni_frames(clip)                 # [T, d_audio] fp32
                tokens = self.resampler(frames.unsqueeze(0))[0]  # [n_query, d_llm]
                out.append(tokens.to(target_dtype))
        return torch.cat(out, dim=0)

    # ------------------------------------------------- token-level adapter gate
    def _audio_row_indices(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Flattened rows (in this extend batch) belonging to audio requests.

        The served audio prompt is exactly ``[audio_pad] * n_query + [im_end]``
        (plus at most a tokenizer-appended trailing pad, which is causally inert for
        the pooled position), so the whole request runs under the audio gate — the
        same readout the reference ``encode_audio`` produces. Placeholder token ids
        are not usable here (SGLang replaces them with per-item hash values), so the
        gate keys off ``forward_batch.mm_inputs`` instead.
        """
        if forward_batch.forward_mode.is_decode():
            return None
        mm_list = forward_batch.mm_inputs
        seq_lens = forward_batch.extend_seq_lens_cpu
        if not mm_list or seq_lens is None:
            return None
        spans: List[Tuple[int, int]] = []
        start = 0
        for i, sl in enumerate(seq_lens):
            sl = int(sl)
            mm = mm_list[i] if i < len(mm_list) else None
            if mm is not None and any(it.is_audio() for it in mm.mm_items):
                spans.append((start, start + sl))
            start += sl
        if not spans:
            return None
        idx = torch.cat([torch.arange(s, e, dtype=torch.long) for s, e in spans])
        return idx.to(forward_batch.input_ids.device)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch,
                get_embedding: bool = False, **kwargs):
        if self.audio_adapters is not None:
            self._adapter_state.idx = self._audio_row_indices(forward_batch)
        # Snapshot per-request modality NOW: general_mm_embed_routine nulls
        # forward_batch.mm_inputs after injecting the embeddings, so the pooler
        # cannot read it at pooling time.
        self.pooler.snapshot_batch_modalities(forward_batch)
        try:
            return super().forward(
                input_ids, positions, forward_batch,
                get_embedding=get_embedding, **kwargs,
            )
        finally:
            self._adapter_state.clear()
            self.pooler.clear_batch_modalities()

    # -------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        from safetensors.torch import safe_open

        # 1) The fusion repo's checkpoint: trained connector + whitening + adapters.
        own_sd: dict[str, torch.Tensor] = {}
        for name, w in weights:
            if name.startswith(("resampler.", "text_whitening.", "audio_adapters.")):
                own_sd[name] = w.clone()
            # logit_scale (training temperature) is unused at inference.

        def _load_into(prefix: str, module: nn.Module) -> None:
            sd = {k[len(prefix):]: v for k, v in own_sd.items() if k.startswith(prefix)}
            if not sd:
                raise RuntimeError(f"checkpoint carries no '{prefix}*' tensors")
            module.load_state_dict(sd, strict=True)

        _load_into("resampler.", self.resampler)
        _load_into("text_whitening.", self.text_whitening)
        if self.audio_adapters is not None:
            _load_into("audio_adapters.", self.audio_adapters)
        if int(self.text_whitening.fitted) != 1:
            raise RuntimeError("text_whitening loaded but not fitted — wrong checkpoint?")

        # 2) The FROZEN base weights from the base repo.
        from huggingface_hub import snapshot_download

        base_snap = snapshot_download(
            self.fusion_base_model, allow_patterns=["*.safetensors", "*.json"]
        )
        base_files = sorted(glob.glob(os.path.join(base_snap, "*.safetensors")))
        if not base_files:
            raise RuntimeError(f"no safetensors found in {base_snap}")

        def _base_iter():
            for shard in base_files:
                with safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("lm_head."):
                            continue
                        yield key, f.get_tensor(key)

        Qwen3VLForConditionalGeneration.load_weights(self, _base_iter())

        # 3) The FROZEN audio tower from the Omni repo (thinker.audio_tower.* only).
        if self.audio_tower is not None:
            omni_snap = snapshot_download(
                self.fusion_audio_model, allow_patterns=["*.safetensors", "*.json"]
            )
            prefix = "thinker.audio_tower."
            collected: dict[str, torch.Tensor] = {}
            for shard in sorted(glob.glob(os.path.join(omni_snap, "*.safetensors"))):
                with safe_open(shard, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith(prefix):
                            collected[key[len(prefix):]] = f.get_tensor(key)
            missing, unexpected = self.audio_tower.load_state_dict(collected, strict=False)
            logger.info(
                "[fusion] audio tower: loaded %d tensors (missing=%d unexpected=%d)",
                len(collected), len(missing), len(unexpected),
            )
            if len(collected) < 400:
                raise RuntimeError(
                    f"audio tower load looks wrong: only {len(collected)} tensors"
                )


EntryClass = FusionEmbeddingModel
