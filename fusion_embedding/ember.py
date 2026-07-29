"""Ember — thermal-infrared image embeddings from a sense pack on fusion-embedding-2.

Ember is the released FE2 base plus a thermal :class:`AdapterPacks` pack (HF repo
``EximiusLabs/fusion-embedding-2-ember``, pack rank 384). It reuses the light
:class:`FusionTextEmbedder` load path (frozen Qwen3-VL-Embedding base, no audio tower)
and attaches the thermal adapter pack to the base decoder layers.

Unlike the whitened 1024-d FE2 retrieval space, Ember's thermal vectors live in the raw
LLM hidden dim (``cfg.d_llm``), L2-normalized. Thermal encodes run with the pack gate
OPEN (``packs.scope("thermal")``); the paired text/caption path runs with the gate CLOSED,
exactly as trained. The math replicates ``scripts/thermal_pack_hub_smoke.py`` bit-for-bit.

    from fusion_embedding import EmberEmbedder
    m = EmberEmbedder.from_pretrained()
    v_img = m.embed_thermal("frame.png")                 # thermal image -> [d_llm], L2
    v_txt = m.embed_text("a person crossing a dark road")# caption       -> [d_llm], L2
    sim = float(v_img @ v_txt)

Requires: torch, transformers>=4.46, huggingface_hub, safetensors.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

THERMAL_INSTRUCTION = "Represent this thermal infrared image."
DOC_INSTRUCTION = "Represent the user's input."
PACK_NAME = "thermal"
PACK_RANK = 384
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
MAX_PIXELS = 1310720   # training-protocol cap (matches the smoke)


def _chat(instruction: str, user: str) -> str:
    """The base's official embedding format: system-turn instruction, assistant opener."""
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


class EmberEmbedder:
    """Thermal + caption encoder: FE2 light base with a thermal adapter pack attached."""

    def __init__(self, fe2_repo_or_path: str = "EximiusLabs/fusion-embedding-2-2b-preview",
                 pack_repo: str = "EximiusLabs/fusion-embedding-2-ember",
                 device: str = "cuda", revision: Optional[str] = None,
                 pack_revision: str = "main", dtype=torch.bfloat16, **kw):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from fusion_embedding.adapters import AdapterPacks
        from fusion_embedding.model import last_token_pool
        from fusion_embedding.text_embedder import FusionTextEmbedder

        self.device = device
        self._pool = last_token_pool

        # Reuse the light base load (base AutoModel + processor + FusionEmbeddingModel with
        # audio_encoder=None, resampler/adapters/text_whitening from the FE2 checkpoint).
        base = FusionTextEmbedder.from_pretrained(fe2_repo_or_path, device=device,
                                                  revision=revision, dtype=dtype, **kw)
        self.base = base
        self.full = base.full
        self.proc = base.proc
        self.tok = base.tok
        self.model = base.model
        self.cfg = base.cfg

        # Training-protocol pixel cap for the vision tower (matches the smoke).
        ip = getattr(self.proc, "image_processor", None)
        if ip is not None and hasattr(ip, "max_pixels"):
            ip.max_pixels = MAX_PIXELS

        # Download and attach the thermal pack.
        st_path = hf_hub_download(pack_repo, "model.safetensors", revision=pack_revision)
        hf_hub_download(pack_repo, "config.json", revision=pack_revision)  # presence-checked artifact
        sd = load_file(st_path)

        self.packs = AdapterPacks()
        adapters, _ = self.packs.add_pack(PACK_NAME, self.model.base_lm,
                                          self.cfg.d_llm, PACK_RANK)
        adapters.load_state_dict(sd)          # strict by default
        self.packs.to(device)

    @classmethod
    def from_pretrained(cls, fe2_repo: str = "EximiusLabs/fusion-embedding-2-2b-preview",
                        pack_repo: str = "EximiusLabs/fusion-embedding-2-ember",
                        device: str = "cuda", revision: Optional[str] = None,
                        pack_revision: str = "main", **kw) -> "EmberEmbedder":
        """Construct from the FE2 base repo and the Ember pack repo (or local FE2 path)."""
        return cls(fe2_repo_or_path=fe2_repo, pack_repo=pack_repo, device=device,
                   revision=revision, pack_revision=pack_revision, **kw)

    @torch.no_grad()
    def embed_thermal(self, image) -> torch.Tensor:
        """Embed one thermal image (PIL.Image or path); L2-normalized [d_llm] vector on CPU.

        Runs with the thermal pack gate OPEN, matching the smoke's ``embed_thermal(open_gate=True)``.
        """
        from PIL import Image

        if isinstance(image, (str, os.PathLike)):
            image = Image.open(str(image))
        image = image.convert("RGB")
        text = _chat(THERMAL_INSTRUCTION, IMAGE_PLACEHOLDER)
        inp = self.proc(text=[text], images=[image], return_tensors="pt").to(self.device)
        with self.packs.scope(PACK_NAME):
            h = self.full(**inp).last_hidden_state
        vec = F.normalize(self._pool(h, inp["attention_mask"]).float(), dim=-1)
        return vec.squeeze(0).cpu()

    @torch.no_grad()
    def embed_text(self, caption: str) -> torch.Tensor:
        """Embed one caption; L2-normalized [d_llm] vector on CPU.

        Runs the frozen base LM over the instruction+text side with the gate CLOSED
        (not wrapped in ``packs.scope``), matching the smoke's ``embed_caps``.
        """
        enc = self.proc.tokenizer(_chat(DOC_INSTRUCTION, caption), return_tensors="pt",
                                  truncation=True, max_length=512).to(self.device)
        tok_emb = self.full.get_input_embeddings()
        o = self.model.base_lm(inputs_embeds=tok_emb(enc["input_ids"]),
                               attention_mask=enc["attention_mask"])
        h = o if isinstance(o, torch.Tensor) else (
            o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0])
        vec = F.normalize(self._pool(h, enc["attention_mask"]).float(), dim=-1)
        return vec.squeeze(0).cpu()

    @torch.no_grad()
    def encode(self, items: Union[str, "os.PathLike", Sequence],
               kind: str = "thermal") -> np.ndarray:
        """Embed a single item or a sequence; returns numpy ([d_llm] or [N, d_llm]).

        ``kind="thermal"`` embeds thermal images (PIL / paths); ``kind="text"`` embeds captions.
        """
        if kind == "thermal":
            fn = self.embed_thermal
            one = isinstance(items, (str, os.PathLike)) or not isinstance(items, (list, tuple))
        elif kind == "text":
            fn = self.embed_text
            one = isinstance(items, str)
        else:
            raise ValueError(f"kind must be 'thermal' or 'text', got {kind!r}")
        seq: List = [items] if one else list(items)
        out = np.vstack([fn(x).numpy() for x in seq])
        return out[0] if one else out
