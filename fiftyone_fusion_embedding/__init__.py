"""Remote zoo model source for the fusion-embedding family.

Register it once, then the models below behave like any other zoo model:

    import fiftyone.zoo as foz
    foz.register_zoo_model_source("https://github.com/Eximius-Labs/fusion-embedding")
    model = foz.load_zoo_model("eximius/tactus-mat")

Naming a model by string (rather than passing an instance) is what lets a similarity
index answer text queries in the App and in later sessions, so always pass the name to
``compute_similarity(model=...)``.
"""
import logging

logger = logging.getLogger(__name__)

MODELS = {
    "eximius/fusion-embedding-2": ("image", {}),
    "eximius/ember": ("image", {"thermal": True}),
    "eximius/tactus": ("sensor", {}),
    "eximius/tactus-mat": ("sensor", {}),
}


def download_model(model_name, model_path):
    """No-op: weights are pulled from the Hub on first use and cached there.

    FiftyOne calls this before ``load_model``. The family's weights live on the
    HuggingFace Hub and each pack downloads its own checkpoint at load time, so there is
    nothing to place at ``model_path``. Validating the name here keeps a typo from
    surfacing later as a confusing load error.
    """
    if model_name not in MODELS:
        raise ValueError(
            f"unsupported model {model_name!r}; choose from {sorted(MODELS)}")

    # The weights live on the HuggingFace Hub and each pack fetches its own checkpoint at
    # load time, so there is nothing large to place here. FiftyOne still expects the file
    # named by ``base_filename`` to exist once a model is "downloaded", so write a marker.
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "w") as f:
        f.write(model_name + ": weights are pulled from the HuggingFace Hub on load")
    logger.info("%s will download its weights from the HuggingFace Hub on first use",
                model_name)


def load_model(model_name, model_path, device=None, **kwargs):
    """Loads the model.

    Args:
        model_name: the name declared in ``manifest.json``
        model_path: unused; the weights come from the Hub
        device: "cuda", "cpu", or None to pick automatically

    Returns:
        a :class:`fiftyone.core.models.Model` that also embeds text prompts
    """
    if model_name not in MODELS:
        raise ValueError(
            f"unsupported model {model_name!r}; choose from {sorted(MODELS)}")

    from .fusion_zoo import FusionImageModel, FusionSensorModel

    kind, defaults = MODELS[model_name]
    opts = {**defaults, **kwargs}
    if kind == "image":
        return FusionImageModel(device=device, **opts)
    return FusionSensorModel(model_name, device=device, **opts)
