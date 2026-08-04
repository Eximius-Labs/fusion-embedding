"""FusionEmbeddingModel — vLLM pooling implementation of fusion-embedding-2.

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

Audio is served as a vLLM multimodal input: the prompt carries ONE
``<|vision_pad|>`` placeholder per clip (token id 151654 — the base tokenizer has no
``<|audio_pad|>`` string; the released checkpoint reuses this inert token as the
audio slot marker). The processor expands it to ``n_query`` (64) slots, the audio
tower + resampler produce the 64 injected embeddings, and the client appends
``<|im_end|>`` so the pooled position matches the reference readout
(``[audio_pad] * 64 + [eos]``).

Gating in a continuously-batched engine cannot be a global on/off switch, so the
adapter hooks are gated PER TOKEN: ``embed_input_ids`` marks the flattened token rows
that belong to audio sequences (the 64 slots plus the trailing ``<|im_end|>``) and the
decoder-layer hooks touch only those rows (index_copy on the masked rows; rows of
non-audio sequences are never read or written, preserving the family's byte-identity
guarantee for text/image/video). The plugin forces eager execution so these Python
hooks can never be baked into a compiled graph.
"""

from __future__ import annotations

import functools
import glob
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.common import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import LastPool
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.adapters import as_embedding_model
from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import PromptReplacement
from vllm.multimodal.processing.context import InputProcessingContext

logger = init_logger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_AUDIO_MODEL = "Qwen/Qwen2.5-Omni-7B"

# The base tokenizer string whose id (151654) the released checkpoint reuses as the
# audio placeholder ("<|audio_pad|>" does not exist in the base vocabulary).
AUDIO_PLACEHOLDER = "<|vision_pad|>"

_ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


def _audio_enabled() -> bool:
    return os.environ.get("FUSION_VLLM_DISABLE_AUDIO", "0") != "1"


def _adapters_enabled() -> bool:
    """Stage-gating escape hatch for the smoke test: FUSION_VLLM_DISABLE_ADAPTERS=1
    skips the gated adapters entirely (frozen-base-only, stage A). The audio path
    REQUIRES the adapters (the released checkpoint was trained with them), so audio
    refuses to run with adapters disabled."""
    return os.environ.get("FUSION_VLLM_DISABLE_ADAPTERS", "0") != "1"


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
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
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
# Token-gated adapter hooks on vLLM decoder layers
# --------------------------------------------------------------------------- #
class _AdapterState:
    """Per-step flattened-row indices of tokens belonging to audio sequences."""

    __slots__ = ("idx",)

    def __init__(self) -> None:
        self.idx: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.idx = None


def _make_layer_hook(adapter: GatedAdapter, state: _AdapterState):
    """Forward hook for a vLLM Qwen3 decoder layer.

    vLLM layers return ``(mlp_delta, residual)`` — the full residual-stream hidden
    state after layer i is ``mlp_delta + residual`` (fused into the next layer's
    input layernorm). The reference applies ``h + adapter(h)`` to the full stream, so
    here: ``full = delta + residual;  delta' = delta + adapter(full)`` which yields
    ``full' = full + adapter(full)`` downstream. Only rows in ``state.idx`` (audio
    sequences) are read or written; with no audio rows the hook is a pure no-op, so
    non-audio forwards remain bitwise those of the frozen base.
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
        full = sub + residual.index_select(0, use)
        hidden.index_copy_(0, use, sub + adapter(full))
        return None

    return hook


# --------------------------------------------------------------------------- #
# Multimodal processing (image/video from Qwen3VL + our audio path)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=2)
def _get_audio_feature_extractor(audio_model: str):
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(audio_model)


class FusionProcessingInfo(Qwen3VLProcessingInfo):
    """Qwen3VL processing info with the HF processor redirected to the BASE repo
    (the fusion repo has no tokenizer/processor files) plus audio support."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        hf_config = ctx.model_config.hf_config
        base_model = getattr(hf_config, "base_model", DEFAULT_BASE_MODEL)
        if ctx.model_config.model != base_model:
            import copy

            mc = copy.copy(ctx.model_config)
            mc.model = base_model
            mc.revision = None
            ctx = InputProcessingContext(model_config=mc, tokenizer=ctx.tokenizer)
        super().__init__(ctx)
        self._audio_model = getattr(hf_config, "audio_model", DEFAULT_AUDIO_MODEL)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        limits = {"image": None, "video": None}
        if _audio_enabled():
            limits["audio"] = None
        return limits

    def get_audio_feature_extractor(self):
        return _get_audio_feature_extractor(self._audio_model)

    def get_data_parser(self):
        kwargs: dict[str, Any] = {}
        if _audio_enabled():
            kwargs["target_sr"] = self.get_audio_feature_extractor().sampling_rate
            # librosa's default resampler (used by the reference embed_audio) is
            # soxr_hq; vLLM's soxr method is the same library, so served resampling
            # matches the reference. The pyav default would drift audibly.
            kwargs["audio_resample_method"] = "soxr"
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
            **kwargs,
        )


class FusionDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        text = super().get_dummy_text(mm_counts)
        return text + AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        data = dict(super().get_dummy_mm_data(seq_len, mm_counts, mm_options))
        num_audios = mm_counts.get("audio", 0)
        if num_audios:
            fe = self.info.get_audio_feature_extractor()
            data["audio"] = self._get_dummy_audios(
                length=int(fe.chunk_length * fe.sampling_rate),
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        return data


class FusionMultiModalProcessor(Qwen3VLMultiModalProcessor):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        output = super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        if audios:
            import numpy as np

            fe = self.info.get_audio_feature_extractor()
            # Reference-exact featurization (modeling_fusion_embedding.embed_audio):
            # 30 s mel window, padded to max_length, then trimmed by attention mask.
            feats = fe(
                [np.asarray(a, dtype=np.float32) for a in audios],
                sampling_rate=fe.sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
            )
            output["input_audio_features"] = feats["input_features"]
            output["audio_feature_lens"] = feats["attention_mask"].sum(dim=-1)
        return output

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        cfg = dict(super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs))
        cfg["input_audio_features"] = MultiModalFieldConfig.batched("audio")
        cfg["audio_feature_lens"] = MultiModalFieldConfig.batched("audio")
        return cfg

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        updates = list(
            super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
        )
        hf_config = self.info.get_hf_config()
        audio_pad_id = int(getattr(hf_config, "audio_pad_id", 151654))
        n_query = int(getattr(hf_config, "n_query", 64))
        updates.append(
            PromptReplacement(
                modality="audio",
                target=[audio_pad_id],
                replacement=[audio_pad_id] * n_query,
            )
        )
        return updates


# --------------------------------------------------------------------------- #
# Pooler: last-token pool -> (text only) whitening -> MRL truncate -> L2 norm
# --------------------------------------------------------------------------- #
class FusionPooler(Pooler):
    def __init__(self, whitening: TextWhitening, vision_token_ids: Sequence[int],
                 audio_pad_id: int, trailing_pad_id: int):
        super().__init__()
        self.pooling = LastPool()   # kept for introspection; forward gathers directly
        self.whitening = whitening
        self.trailing_pad_id = int(trailing_pad_id)
        self.head = EmbeddingPoolerHead(
            head_dtype=torch.float32, activation=PoolerNormalize()
        )
        self.register_buffer(
            "_vision_ids",
            torch.tensor(sorted(set(int(i) for i in vision_token_ids)),
                         dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_non_text_ids",
            torch.tensor(sorted({int(audio_pad_id), *(int(i) for i in vision_token_ids)}),
                         dtype=torch.long),
            persistent=False,
        )

    def get_supported_tasks(self):
        return {"embed"}

    def get_pooling_updates(self, task) -> PoolingParamsUpdate:
        # Token ids are needed to decide which sequences are pure text (whitened)
        # and to normalize away a tokenizer-appended trailing <|endoftext|>.
        return PoolingParamsUpdate(requires_token_ids=True)

    def _pool_last_content(self, hidden_states: torch.Tensor, pooling_metadata,
                           token_id_rows) -> torch.Tensor:
        """Last-token pooling that reproduces the released readout per modality.

        The base tokenizer appends ``<|endoftext|>`` (the pad id) when called with
        the default ``add_special_tokens=True``. The reference readout tokenizes
        TEXT with ``add_special_tokens=False`` (pools at the assistant-opener
        newline) but images/videos through the HF processor default (pools AT the
        appended ``<|endoftext|>``); reference audio has no trailing pad (pools at
        ``<|im_end|>``). Attention is causal, so the hidden state one position
        before a trailing pad is bit-identical to a no-pad prompt's last hidden —
        therefore: step back over exactly one trailing pad for sequences WITHOUT
        vision tokens (text and audio); keep the true last token for vision
        sequences.
        """
        cursor = pooling_metadata.get_pooling_cursor()
        last_idx = cursor.last_token_indices_gpu
        n_sched = cursor.num_scheduled_tokens_cpu
        vision_ids = self._vision_ids.to(hidden_states.device)
        offsets = []
        for i, row in enumerate(token_id_rows):
            row_dev = row.to(hidden_states.device)
            step_back = (
                len(row) > 1
                and int(row[-1]) == self.trailing_pad_id
                and int(n_sched[i]) >= 2   # never step across a sequence boundary
                and not bool(torch.isin(row_dev, vision_ids).any())
            )
            offsets.append(1 if step_back else 0)
        if any(offsets):
            last_idx = last_idx - torch.tensor(
                offsets, device=last_idx.device, dtype=last_idx.dtype
            )
        return hidden_states[last_idx]

    def forward(self, hidden_states: torch.Tensor, pooling_metadata):
        token_id_rows = pooling_metadata.get_prompt_token_ids()
        pooled = self._pool_last_content(hidden_states, pooling_metadata, token_id_rows)
        pooled = pooled.to(torch.float32)

        if len(token_id_rows) != pooled.shape[0]:
            raise RuntimeError(
                f"pooled batch ({pooled.shape[0]}) != prompt rows ({len(token_id_rows)})"
            )
        mm_ids = self._non_text_ids.to(pooled.device)
        flags = [
            not bool(torch.isin(row.to(pooled.device), mm_ids).any())
            for row in token_id_rows
        ]
        if any(flags):
            whitened = self.whitening(pooled)
            sel = torch.tensor(flags, device=pooled.device).unsqueeze(-1)
            pooled = torch.where(sel, whitened, pooled)
        # The head applies (optional) matryoshka truncation, then L2 normalization —
        # the same order as the reference contract (whiten raw, truncate, normalize).
        return self.head(pooled, pooling_metadata)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
_Qwen3VLForEmbedding = as_embedding_model(Qwen3VLForConditionalGeneration)


@MULTIMODAL_REGISTRY.register_processor(
    FusionMultiModalProcessor,
    info=FusionProcessingInfo,
    dummy_inputs=FusionDummyInputsBuilder,
)
class FusionEmbeddingModel(_Qwen3VLForEmbedding):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER
        return Qwen3VLForConditionalGeneration.get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = vllm_config.model_config.hf_config

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

        # ---- gated adapters on the vLLM decoder layers ----
        self.audio_adapters: Optional[nn.ModuleList] = None
        self._adapter_state = _AdapterState()
        adapter_rank = int(getattr(cfg, "adapter_rank", 0) or 0)
        if adapter_rank > 0 and not _adapters_enabled():
            logger.warning("[fusion] gated adapters DISABLED via env — frozen-base only; "
                           "the audio path is unavailable in this configuration")
            adapter_rank = 0
        if adapter_rank > 0:
            layers = self.language_model.model.layers
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
            self.audio_tower = Qwen2_5OmniAudioEncoder(audio_cfg).to(
                vllm_config.model_config.dtype
            )
            self.audio_tower.eval()
            if int(getattr(cfg, "adapter_rank", 0) or 0) > 0 and self.audio_adapters is None:
                raise RuntimeError(
                    "the checkpoint's audio path requires its gated adapters; unset "
                    "FUSION_VLLM_DISABLE_ADAPTERS or set FUSION_VLLM_DISABLE_AUDIO=1"
                )

        # ---- pooler: replace the generic embedding pooler with ours ----
        vision_ids = []
        for attr in ("image_token_id", "video_token_id"):
            tid = getattr(cfg, attr, None)
            if tid is not None:
                vision_ids.append(int(tid))
        if not vision_ids:
            raise RuntimeError("merged config carries no image/video token ids")
        self.pooler = FusionPooler(
            self.text_whitening, vision_ids,
            audio_pad_id=self.fusion_audio_pad_id,
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
        assert self.audio_tower is not None, "audio path disabled (FUSION_VLLM_DISABLE_AUDIO)"
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
    def _embed_audio(self, feats, lens) -> list[torch.Tensor]:
        if isinstance(feats, torch.Tensor):
            items = list(feats.unbind(0))
        else:
            items = [f[0] if f.dim() == 3 else f for f in feats]
        if isinstance(lens, torch.Tensor):
            lens_list = [int(x) for x in lens.reshape(-1).tolist()]
        else:
            lens_list = [int(x) for x in lens]
        target_dtype = self.language_model.model.embed_tokens.weight.dtype
        out: list[torch.Tensor] = []
        for mel, L in zip(items, lens_list):
            mel = mel[..., : max(L, 1)]
            frames = self._omni_frames(mel)                    # [T, d_audio] fp32
            tokens = self.resampler(frames.unsqueeze(0))       # [1, n_query, d_llm]
            tokens = tokens[0].to(target_dtype)
            if self.use_deepstack:
                # Vision embeddings carry [visual_dim | deepstack multiscale] channels
                # and embed_input_ids splits them apart. Audio has no deepstack levels;
                # zero multiscale channels reproduce exactly what every non-vision
                # position receives (the deepstack buffer is zero-initialized).
                tokens = F.pad(tokens, (0, self.multiscale_dim))
            out.append(tokens)
        return out

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Qwen3VL's mrope walker only understands image/video grids. Audio slots are
        # position-wise plain text (the reference runs the 64+1 audio tokens with
        # ordinary sequential positions), so drop audio features before delegating —
        # non-vision spans get sequential positions in all three mrope rows.
        non_audio = [f for f in mm_features if f.modality != "audio"]
        return super().get_mrope_input_positions(input_tokens, non_audio)

    def embed_multimodal(self, **kwargs: object):
        if kwargs.get("input_audio_features") is not None:
            return self._embed_audio(
                kwargs["input_audio_features"], kwargs.get("audio_feature_lens")
            )
        return super().embed_multimodal(**kwargs)

    # ------------------------------------------------- token-level adapter gate
    def _update_audio_row_state(self, input_ids: Optional[torch.Tensor]) -> None:
        if self.audio_adapters is None:
            return
        self._adapter_state.clear()
        if input_ids is None:
            return
        is_pad = input_ids == self.fusion_audio_pad_id
        if not bool(is_pad.any()):
            return
        # Audio sequences are exactly [audio_pad] * n_query + [eos]; in the flattened
        # batch that is a run of pads followed by the sequence's own eos. Mark the
        # pads plus an eos that immediately follows a pad.
        follows_pad = torch.zeros_like(is_pad)
        follows_pad[1:] = is_pad[:-1]
        mask = is_pad | (follows_pad & (input_ids == self.fusion_eos_id))
        self._adapter_state.idx = mask.nonzero(as_tuple=False).squeeze(-1)

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, **kwargs):
        self._update_audio_row_state(input_ids)
        return super().embed_input_ids(input_ids, multimodal_embeddings, **kwargs)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        # If the runner hands us raw input_ids (text-only fallback path), the state
        # from embed_input_ids may be stale — recompute from what we actually run.
        if input_ids is not None:
            self._update_audio_row_state(input_ids)
        try:
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        finally:
            self._adapter_state.clear()

    # -------------------------------------------------------------- weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # 1) The fusion repo's checkpoint: trained connector + whitening + adapters.
        own_sd: dict[str, torch.Tensor] = {}
        for name, w in weights:
            if name.startswith(("resampler.", "text_whitening.", "audio_adapters.")):
                own_sd[name] = w.clone()
            # logit_scale (training temperature) is unused at inference.

        loaded: set[str] = set()

        def _load_into(prefix: str, module: nn.Module) -> None:
            sd = {k[len(prefix):]: v for k, v in own_sd.items() if k.startswith(prefix)}
            if not sd:
                raise RuntimeError(f"checkpoint carries no '{prefix}*' tensors")
            module.load_state_dict(sd, strict=True)
            loaded.update(prefix + k for k, _ in module.named_parameters())

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
        base_iter = safetensors_weights_iterator(base_files, use_tqdm_on_load=False)
        filtered = ((n, w) for n, w in base_iter if not n.startswith("lm_head."))
        base_loaded = Qwen3VLForConditionalGeneration.load_weights(self, filtered)
        loaded |= set(base_loaded or ())

        # 3) The FROZEN audio tower from the Omni repo (thinker.audio_tower.* only).
        if self.audio_tower is not None:
            from safetensors.torch import load_file

            omni_snap = snapshot_download(
                self.fusion_audio_model, allow_patterns=["*.safetensors", "*.json"]
            )
            prefix = "thinker.audio_tower."
            collected: dict[str, torch.Tensor] = {}
            for shard in sorted(glob.glob(os.path.join(omni_snap, "*.safetensors"))):
                for k, v in load_file(shard).items():
                    if k.startswith(prefix):
                        collected[k[len(prefix):]] = v
            missing, unexpected = self.audio_tower.load_state_dict(collected, strict=False)
            logger.info(
                "[fusion] audio tower: loaded %d tensors (missing=%d unexpected=%d)",
                len(collected), len(missing), len(unexpected),
            )
            if len(collected) < 400:
                raise RuntimeError(
                    f"audio tower load looks wrong: only {len(collected)} tensors"
                )
            loaded.update("audio_tower." + k for k, _ in self.audio_tower.named_parameters())

        return loaded
