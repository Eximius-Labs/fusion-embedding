"""SGLang out-of-tree plugin for the fusion-embedding-2 family (FusionEmbeddingModel).

The plugin is wired through SGLang's official extension seams:

* ``sglang.srt.plugins`` entry point (this module's :func:`register`) — runs at
  ``load_plugins()`` in EVERY SGLang process (CLI entry, engine, scheduler) BEFORE any
  ``ServerArgs``/``ModelConfig`` is constructed. It registers the shell-config class on
  ``transformers.AutoConfig`` (so the repo parses without ``trust_remote_code``) and
  installs a ``ServerArgs.__post_init__`` hook with the model-specific launch
  adjustments (tokenizer redirect, eager execution, radix cache off).
* ``SGLANG_EXTERNAL_MODEL_PACKAGE=fusion_embedding.sglang_plugin.models`` — the model registry
  imports the package and registers ``FusionEmbeddingModel`` (via ``EntryClass``).
* ``SGLANG_EXTERNAL_MM_MODEL_ARCH=FusionEmbeddingModel`` — marks the arch multimodal.
* ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=fusion_embedding.sglang_plugin.processors`` — the
  tokenizer manager imports the package and registers the multimodal processor.

:func:`register` defaults the three env vars (``os.environ.setdefault``) so a plain
launch works once the package is pip-installed; exporting them explicitly (the
documented launch contract) is always honored.

Launch contract (pinned against sglang==0.5.16)::

    SGLANG_EXTERNAL_MODEL_PACKAGE=fusion_embedding.sglang_plugin.models \
    SGLANG_EXTERNAL_MM_MODEL_ARCH=FusionEmbeddingModel \
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=fusion_embedding.sglang_plugin.processors \
    python -m sglang.launch_server \
        --model-path EximiusLabs/fusion-embedding-2-2b-preview --is-embedding
"""

import logging
import os

logger = logging.getLogger(__name__)

FUSION_MODEL_TYPE = "fusion-embedding-connector"
FUSION_ARCH = "FusionEmbeddingModel"

_EXTERNAL_ENV_DEFAULTS = {
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "fusion_embedding.sglang_plugin.models",
    "SGLANG_EXTERNAL_MM_MODEL_ARCH": FUSION_ARCH,
    "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "fusion_embedding.sglang_plugin.processors",
}

_registered = False


def _server_args_post_init_hook(result, server_args, *args, **kwargs):
    """AFTER hook on ``ServerArgs.__post_init__`` — the SGLang equivalent of the vLLM
    plugin's per-architecture config hook.

    Only touches launches whose model resolves to the fusion connector:

    * tokenizer/processor redirect to the frozen base repo (the fusion repo carries no
      tokenizer/processor files),
    * the token-gated adapter hooks are Python forward hooks and must never be baked
      into a captured/compiled graph -> CUDA graphs off, torch.compile off,
    * radix prefix cache off and chunked prefill off: the readout contract pools with a
      modality-aware step-back over a trailing pad, which requires whole sequences in
      one extend and no cross-request prefix reuse.
    """
    try:
        hf_config = server_args.get_model_config().hf_config
    except Exception:
        return None
    if getattr(hf_config, "model_type", None) != FUSION_MODEL_TYPE:
        return None

    if not server_args.is_embedding:
        raise ValueError(
            "FusionEmbeddingModel is an embedding model; launch with --is-embedding"
        )

    base_model = getattr(hf_config, "base_model", "Qwen/Qwen3-VL-Embedding-2B")
    if server_args.tokenizer_path in (None, server_args.model_path):
        server_args.tokenizer_path = base_model
        logger.info("[fusion] tokenizer_path -> %s (fusion repo has no tokenizer)",
                    base_model)

    server_args.disable_radix_cache = True
    server_args.chunked_prefill_size = -1
    server_args.disable_cuda_graph = True
    if str(server_args.dtype) in ("float32", "float"):
        # The sgl-kernel fused ops (rmsnorm, silu_and_mul) are half-precision only;
        # fp32 (the structural parity precision) needs the native torch paths. The
        # env is read at plugin load in the scheduler subprocess, which inherits it.
        os.environ.setdefault("FUSION_SGLANG_NATIVE_OPS", "1")
        logger.info("[fusion] dtype=float32: forcing native torch ops "
                    "(FUSION_SGLANG_NATIVE_OPS=1)")
    if getattr(server_args, "enable_torch_compile", False):
        server_args.enable_torch_compile = False
        logger.warning("[fusion] torch.compile disabled: the token-gated adapter "
                       "hooks must run eagerly")
    try:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        server_args.cuda_graph_config.decode.backend = Backend.DISABLED
        server_args.cuda_graph_config.prefill.backend = Backend.DISABLED
    except Exception:
        # Older/newer layouts without cuda_graph_config: disable_cuda_graph covers it.
        pass
    logger.info("[fusion] launch adjustments applied: eager execution, radix cache "
                "off, chunked prefill off")
    return None


def _force_native_ops() -> None:
    """Swap the sgl-kernel fused ops for their reference torch implementations.

    Needed for fp32 serving (the parity-gate precision): the fused rmsnorm /
    silu_and_mul kernels dispatch half dtypes only. ``MultiPlatformOp`` binds
    ``forward_cuda`` per instance at construction, so patching the class before the
    model is built (plugin load precedes model load in the scheduler) is effective.
    """
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm

    RMSNorm.forward_cuda = RMSNorm.forward_native
    SiluAndMul.forward_cuda = SiluAndMul.forward_native
    logger.info("[fusion] native torch ops active (RMSNorm, SiluAndMul)")


def register() -> None:
    """``sglang.srt.plugins`` entry point. Must be safe to call multiple times, from
    any process, and must not initialize CUDA."""
    global _registered
    if _registered:
        return
    _registered = True

    # 1) Teach transformers our shell config type so AutoConfig resolves it locally.
    from transformers import AutoConfig

    from .hf_config import FusionEmbeddingHFConfig

    AutoConfig.register(FUSION_MODEL_TYPE, FusionEmbeddingHFConfig, exist_ok=True)

    # 2) Default the external-package env hooks (explicit exports always win). These
    #    are read later, at model-registry / tokenizer-manager import time, and are
    #    inherited by the scheduler subprocess.
    for key, value in _EXTERNAL_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    # 3) fp32 parity mode: replace half-only fused kernels with native torch ops.
    if os.environ.get("FUSION_SGLANG_NATIVE_OPS", "0") == "1":
        try:
            _force_native_ops()
        except Exception:
            logger.exception("[fusion] failed to force native ops")

    # 4) Model-specific ServerArgs adjustments (no-op for every other model).
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register(
            "sglang.srt.server_args.ServerArgs.__post_init__",
            _server_args_post_init_hook,
            HookType.AFTER,
        )
    except Exception:  # pragma: no cover - hook registry unavailable
        logger.exception("[fusion] could not register the ServerArgs hook; pass "
                         "--tokenizer-path Qwen/Qwen3-VL-Embedding-2B "
                         "--disable-radix-cache --chunked-prefill-size -1 "
                         "--disable-cuda-graph manually")
