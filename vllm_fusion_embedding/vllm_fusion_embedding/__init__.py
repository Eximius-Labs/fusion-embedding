"""vLLM out-of-tree plugin for the fusion-embedding-2 family (FusionEmbeddingModel).

Registered through the ``vllm.general_plugins`` entry point (see pyproject.toml), so a
plain

    vllm serve EximiusLabs/fusion-embedding-2-2b-preview --runner pooling

resolves the repo's ``architectures: ["FusionEmbeddingModel"]`` to the pooling model
implemented in :mod:`vllm_fusion_embedding.model`.

``register()`` must be safe to call multiple times and from any process (API server,
engine core, workers) and must not initialize CUDA.
"""

FUSION_MODEL_TYPE = "fusion-embedding-connector"
FUSION_ARCH = "FusionEmbeddingModel"

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    # 1) Teach transformers our shell config type so AutoConfig resolves it locally
    #    (no trust_remote_code needed; the HF repo's auto_map is ignored).
    from transformers import AutoConfig

    from .hf_config import FusionEmbeddingHFConfig

    AutoConfig.register(FUSION_MODEL_TYPE, FusionEmbeddingHFConfig, exist_ok=True)

    # 2) Register the model architecture (lazy import string: avoids CUDA init here).
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        FUSION_ARCH, "vllm_fusion_embedding.model:FusionEmbeddingModel"
    )

    # 3) Per-architecture config hook: redirect the tokenizer to the frozen base repo
    #    (the fusion repo carries no tokenizer/processor files) and force eager
    #    execution (the gated adapter hooks must never be baked into a compiled or
    #    CUDA-graph-captured forward).
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, VerifyAndUpdateConfig

    class FusionEmbeddingModelConfigHook(VerifyAndUpdateConfig):
        @staticmethod
        def verify_and_update_model_config(model_config) -> None:
            hf_config = model_config.hf_config
            base_model = getattr(hf_config, "base_model", "Qwen/Qwen3-VL-Embedding-2B")
            if model_config.tokenizer in (None, model_config.model):
                model_config.tokenizer = base_model
            if not model_config.enforce_eager:
                model_config.enforce_eager = True

    MODELS_CONFIG_MAP[FUSION_ARCH] = FusionEmbeddingModelConfigHook
