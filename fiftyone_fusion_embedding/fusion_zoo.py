"""FiftyOne model wrappers for the fusion-embedding family.

Each wrapper implements FiftyOne's ``Model`` interface plus ``EmbeddingsMixin`` and
``PromptMixin``, so a similarity index built with one of these supports natural-language
queries in the App and in later Python sessions.

Two kinds of model live here:

``FusionImageModel``
    Reads image media directly (the base model's vision path, and Ember's thermal path).
    Works with ``fob.compute_similarity(dataset, model="eximius/...")`` out of the box.

``FusionSensorModel``
    Reads a sensor stream that FiftyOne does not treat as media: pressure arrays, inertial
    windows, audio. FiftyOne's ``Model.media_type`` currently admits only "image" and
    "video", so these are used through the plugin's operator, which embeds a *field* and
    hands the vectors to ``compute_similarity(embeddings=...)`` while naming this model for
    prompt support. See EMBEDDING SENSOR FIELDS in the README.

Every wrapper embeds text through the family's canonical whitened readout, which is the
same readout each sense pack was trained against. Embedding prompts any other way, including
with the plain base model, misranks.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

BASE_REPO = "EximiusLabs/fusion-embedding-2-2b-preview"

# name -> (sense pack repo or None for the base, revision, sensor grid, doc)
SENSOR_PACKS = {
    "eximius/tactus": (
        "EximiusLabs/fusion-embedding-2-tactus", "v0.1-preview", (32, 32),
        "touch, 32x32 pressure glove",
    ),
    "eximius/tactus-mat": (
        "EximiusLabs/fusion-embedding-2-tactus-mat", "v0.1-preview", (64, 32),
        "touch, 64x32 body pressure mat",
    ),
}


def _torch():
    import torch
    return torch


class _FusionBase:
    """Shared text side: the canonical whitened readout of the frozen base."""

    def __init__(self, device: str | None = None, dtype=None):
        torch = _torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.device = device
        self._dtype = dtype
        self._ue = None

    @property
    def ue(self):
        """The base embedder, loaded lazily so importing this module is cheap."""
        if self._ue is None:
            from fusion_embedding import UnifiedEmbedder
            self._ue = UnifiedEmbedder.from_pretrained(
                BASE_REPO, device=self.device, dtype=self._dtype)
        return self._ue

    # ---- Model interface ----
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def media_type(self):
        return "image"

    @property
    def has_logits(self):
        return False

    @property
    def has_embeddings(self):
        return True

    @property
    def can_embed_prompts(self):
        return True

    @property
    def ragged_batches(self):
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return False

    @preprocess.setter
    def preprocess(self, value):
        pass

    # ---- PromptMixin ----
    def embed_prompt(self, arg):
        return self.embed_prompts([arg])[0]

    def embed_prompts(self, args):
        torch = _torch()
        import torch.nn.functional as F
        with torch.no_grad():
            rows = [self.ue.embed_text(str(a)).float() for a in args]
        return F.normalize(torch.stack(rows), dim=-1).cpu().numpy()


class FusionImageModel(_FusionBase):
    """Image or thermal-image embeddings in the shared space."""

    def __init__(self, thermal: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.thermal = bool(thermal)

    def predict(self, arg):
        self._last = self.embed(arg)
        return None

    def get_embeddings(self):
        return self._last[np.newaxis, :]

    def embed(self, arg):
        torch = _torch()
        import torch.nn.functional as F
        from PIL import Image
        img = arg
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        with torch.no_grad():
            v = self.ue.embed_image(img).float()
        return F.normalize(v, dim=-1).cpu().numpy()

    def embed_all(self, args):
        return np.stack([self.embed(a) for a in args])


class FusionSensorModel(_FusionBase):
    """A sense pack: pressure arrays today, the same shape for other streams.

    ``embed_sensor`` is the entry point used by the plugin operator. ``embed`` is
    implemented so the object still satisfies ``EmbeddingsMixin`` if something calls it
    with an already-shaped array.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)
        if model_name not in SENSOR_PACKS:
            raise ValueError(
                f"unknown sensor model {model_name!r}; "
                f"choose from {sorted(SENSOR_PACKS)}")
        self.model_name = model_name
        repo, revision, grid, _doc = SENSOR_PACKS[model_name]
        self.repo, self.revision, self.grid = repo, revision, grid
        self._head = None

    @property
    def head(self):
        """The sense pack's own inference class, loaded from its Hub repo."""
        if self._head is None:
            import importlib.util
            import os
            import sys

            from huggingface_hub import hf_hub_download
            for f in ("inference.py", "tactile.py"):
                path = hf_hub_download(self.repo, f, revision=self.revision)
            sys.path.insert(0, os.path.dirname(path))
            spec = importlib.util.spec_from_file_location(
                f"fusion_pack_{self.model_name.replace('/', '_').replace('-', '_')}",
                hf_hub_download(self.repo, "inference.py", revision=self.revision))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = getattr(mod, "TactusMatEmbedder", None) or mod.TactusEmbedder
            # the pack loads its own text side; we already hold one, so skip it
            self._head = cls.from_pretrained(
                self.repo, revision=self.revision, device=self.device, load_text=False)
        return self._head

    def embed_sensor(self, window) -> np.ndarray:
        """A sensor window -> one L2-normalized vector in the shared space.

        ``window`` is ``[F, H, W]`` or a single ``[H, W]`` frame, as floats in [0, 1] or
        raw counts (the pack's own preprocessing handles the scaling).
        """
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.shape[-2:] != self.grid:
            raise ValueError(
                f"{self.model_name} expects a {self.grid[0]}x{self.grid[1]} grid, "
                f"got {tuple(arr.shape[-2:])}")
        return self.head.embed_pressure(arr).numpy()

    def embed_sensors(self, windows) -> np.ndarray:
        return np.stack([self.embed_sensor(w) for w in windows])

    def predict(self, arg):
        self._last = self.embed(arg)
        return None

    def get_embeddings(self):
        return self._last[np.newaxis, :]

    def embed(self, arg):
        return self.embed_sensor(arg)

    def embed_all(self, args):
        return self.embed_sensors(args)
