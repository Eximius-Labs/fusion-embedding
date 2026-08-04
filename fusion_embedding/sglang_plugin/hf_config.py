"""HF config bridge: the fusion repo's shell config -> a full Qwen3VL-shaped config.

The EximiusLabs fusion-embedding-2 repo's ``config.json`` describes ONLY the trained
connector (resampler dims, adapter rank, token ids, ...) plus the names of the frozen
component repos. SGLang, however, sizes the KV cache / attention / decoder from the HF
config, so this class merges the shell config with the frozen base's own config
(downloaded from ``base_model``) at parse time. The result is a
``Qwen3VLConfig``-compatible object whose ``model_type`` stays
``fusion-embedding-connector`` and whose ``architectures`` stay
``["FusionEmbeddingModel"]``, with every connector field kept as an extra attribute.

Registered on ``transformers.AutoConfig`` by :func:`fusion_embedding.sglang_plugin.register`
(the ``sglang.srt.plugins`` entry point), so no ``trust_remote_code`` is needed and the
HF repo's ``auto_map`` is ignored.
"""

from __future__ import annotations

from transformers import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"

# Connector fields we expect on the shell config (kept for documentation; anything
# else the shell declares is carried over as well).
_CONNECTOR_KEYS = (
    "d_audio", "d_llm", "n_query", "d_resampler", "resampler_depth",
    "resampler_heads", "resampler_ffn_mult", "resampler_dropout",
    "adapter_rank", "adapter_act", "mrl_dims", "mrl_default",
    "audio_pad_id", "eos_id", "pad_id", "audio_pad_token",
    "base_model", "audio_model", "max_text_tokens", "n_decoder_layers",
)

# Never carried over from the shell into the merged config.
_DROP_KEYS = {"architectures", "auto_map", "model_type", "transformers_version",
              "torch_dtype", "dtype"}


def _merge_with_base(shell: dict) -> dict:
    base_model = shell.get("base_model", DEFAULT_BASE_MODEL)
    base_dict, _ = PretrainedConfig.get_config_dict(base_model)
    merged = dict(base_dict)
    merged.update({k: v for k, v in shell.items() if k not in _DROP_KEYS})
    merged["model_type"] = FusionEmbeddingHFConfig.model_type
    merged["architectures"] = ["FusionEmbeddingModel"]
    # Advertise the MRL ladder for matryoshka `dimensions` requests.
    merged.setdefault("matryoshka_dimensions", list(shell.get("mrl_dims", []) or []))
    return merged


class FusionEmbeddingHFConfig(Qwen3VLConfig):
    model_type = "fusion-embedding-connector"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # A shell config (no text_config) comes from the fusion repo: merge in the
        # frozen base's config. A full dict (e.g. round-tripped through to_dict)
        # passes through unchanged.
        if "text_config" not in config_dict:
            config_dict = _merge_with_base(dict(config_dict))
        return super().from_dict(config_dict, **kwargs)
