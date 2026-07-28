"""Lightweight text-only embedding for a fusion-embedding checkpoint.

Loads only the frozen Qwen3-VL-Embedding base and the connector's trained text head
(text whitening + MRL), and skips the audio tower entirely. The text vectors are identical
to ``FusionEmbedder.embed_text`` but the load is far cheaper (no 7B audio encoder), which
makes it practical for retrieval / RAG integrations that only embed text.

    from fusion_embedding import FusionTextEmbedder
    m = FusionTextEmbedder.from_pretrained("EximiusLabs/fusion-embedding-2-2b-preview")
    vecs = m.encode(["a dog on a beach", "a piano melody"])   # np.ndarray [N, dim]

Requires: torch, transformers>=4.46, huggingface_hub.
"""

from __future__ import annotations

import dataclasses
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

BASE_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."
CKPT_FILES = ("fusion-embedding-2-2b-preview.pt", "fusion-embedding-1-2b-preview.pt")


def _chat(instruction: str, user_content: str) -> str:
    """The base's official embedding format: system-turn instruction, assistant opener."""
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


class FusionTextEmbedder:
    """Text encoder over a fusion-embedding checkpoint, without the audio tower."""

    def __init__(self, ckpt_path: str, device: str = "cuda", dtype=torch.bfloat16):
        from transformers import AutoModel, AutoProcessor

        from fusion_embedding.config import FusionConfig
        from fusion_embedding.hf_components import BaseLMAdapter
        from fusion_embedding.model import FusionEmbeddingModel

        self.device = device
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        flds = {f.name for f in dataclasses.fields(FusionConfig)}
        self.cfg = FusionConfig(**{k: v for k, v in ck["config"].items() if k in flds})

        self.full = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True, dtype=dtype)
        self.full = self.full.to(device).eval()
        for p in self.full.parameters():
            p.requires_grad_(False)
        self.proc = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.tok = self.proc.tokenizer

        # audio_encoder=None: the text path never touches it, so we skip the 7B tower.
        self.model = FusionEmbeddingModel(self.cfg, self.full.get_input_embeddings(),
                                          BaseLMAdapter(self.full.language_model),
                                          audio_encoder=None)
        self.model.resampler.to(device).float()
        self.model.resampler.load_state_dict(ck["resampler"])
        # gated adapters (fusion-embedding-2) attach to the decoder but stay closed for text,
        # so text output is unchanged; load them if present to mirror the full model exactly.
        if self.model.audio_adapters is not None and "adapters" in ck:
            self.model.audio_adapters.to(device).float()
            self.model.audio_adapters.load_state_dict(ck["adapters"])
        self.model.text_whitening.load_state_dict(ck["text_whitening"])   # identity if unfitted
        self.model.eval()

    @classmethod
    def from_pretrained(cls, repo_or_path: str, device: str = "cuda",
                        revision: Optional[str] = None, **kw) -> "FusionTextEmbedder":
        """Load from a local checkpoint path or an HF repo (e.g.
        ``EximiusLabs/fusion-embedding-2-2b-preview``). ``revision`` pins a tag/commit."""
        if os.path.exists(repo_or_path):
            path = repo_or_path
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
            path = None
            for name in CKPT_FILES:
                try:
                    path = hf_hub_download(repo_or_path, name, revision=revision)
                    break
                except EntryNotFoundError:
                    continue
            if path is None:
                raise FileNotFoundError(f"no known checkpoint file in {repo_or_path} "
                                        f"(looked for {CKPT_FILES})")
        return cls(path, device=device, **kw)

    @torch.no_grad()
    def embed_text(self, text: str, instruction: str = DEFAULT_QUERY_INSTRUCTION,
                   dim: Optional[int] = None) -> torch.Tensor:
        """Embed one string; returns an L2-normalized vector on CPU."""
        from fusion_embedding.model import mrl_truncate_normalize
        ids = self.tok.encode(_chat(instruction, text), add_special_tokens=False)[:512]
        ids_t = torch.tensor([ids], device=self.device)
        pooled = self.model.encode_text(ids_t, torch.ones_like(ids_t))
        pooled = self.model.text_whitening(pooled)
        return mrl_truncate_normalize(pooled.float(), dim or self.cfg.mrl_default).squeeze(0).cpu()

    @torch.no_grad()
    def encode(self, texts: Union[str, Sequence[str]],
               instruction: str = DEFAULT_QUERY_INSTRUCTION,
               dim: Optional[int] = None) -> np.ndarray:
        """Embed a string or list of strings; returns a numpy array ([dim] or [N, dim])."""
        one = isinstance(texts, str)
        arr: List[str] = [texts] if one else list(texts)
        out = np.vstack([self.embed_text(t, instruction, dim).numpy() for t in arr])
        return out[0] if one else out
