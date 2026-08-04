"""Multimodal processor for FusionEmbeddingModel (image/video from Qwen3VL + audio).

Registered through ``SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE``. Inherits the stock
Qwen-VL processor for images/videos (HF processor of the frozen base repo — the
plugin's ServerArgs hook redirects ``tokenizer_path`` there) and adds the audio path:

* the prompt carries ONE ``<|vision_pad|>`` (id 151654) per clip; after HF
  processing this module expands it to ``n_query`` (64) slots so the offsets and
  M-RoPE positions line up with the injected resampler tokens;
* featurization is reference-exact (``modeling_fusion_embedding.embed_audio``):
  Whisper 30 s mel window padded to max_length, attention-mask lengths attached,
  trimming happens model-side;
* audio decoding/resampling is done HERE with soundfile + librosa (soxr_hq — what
  training used) instead of SGLang's default torchcodec/torchaudio loader, whose
  resampler drifts audibly from the reference.

The parent processor is constructed with a shallow config copy whose ``model_type``
is ``qwen3_vl`` so every inherited branch (video handling, ``get_rope_index``)
follows the Qwen3-VL semantics of the frozen base; audio slots carry no vision
markers, so ``get_rope_index`` gives them plain sequential positions in all three
rope rows — the reference behavior.
"""

from __future__ import annotations

import copy
import functools
import io
import re
from typing import Optional

import numpy as np

from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.managers.schedule_batch import Modality

from fusion_embedding.sglang_plugin.models.fusion_embedding import (
    AUDIO_PLACEHOLDER,
    DEFAULT_AUDIO_MODEL,
    FusionEmbeddingModel,
)

WHISPER_SR = 16000


@functools.lru_cache(maxsize=2)
def _get_audio_feature_extractor(audio_model: str):
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(audio_model)


def _decode_audio_reference(data, target_sr: int = WHISPER_SR) -> np.ndarray:
    """Decode + resample audio exactly like the reference ``embed_audio``:
    soundfile decode, channel mean, librosa resample (soxr_hq, librosa>=0.10
    default). Accepts a file path, ``file://``/``http(s)://`` URL, ``data:`` base64
    URL, or raw bytes."""
    import soundfile as sf

    if isinstance(data, str):
        if data.startswith("data:"):
            import base64

            data = base64.b64decode(data.split(",", 1)[1])
        elif data.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(data, timeout=30) as resp:
                data = resp.read()
        elif data.startswith("file://"):
            from urllib.parse import unquote, urlparse

            data = unquote(urlparse(data).path)
    if isinstance(data, bytes):
        wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    else:
        wav, sr = sf.read(str(data), dtype="float32")
    if getattr(wav, "ndim", 1) > 1:
        wav = wav.mean(axis=1)
    if int(sr) != int(target_sr):
        import librosa

        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr))
    return np.asarray(wav, dtype=np.float32)


class FusionMultimodalProcessor(QwenVLImageProcessor):
    models = [FusionEmbeddingModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # The inherited machinery (video branch, get_rope_index) keys off
        # model_type; hand it a copy that reads as the frozen base's qwen3_vl.
        cfg = copy.copy(hf_config)
        cfg.model_type = "qwen3_vl"
        super().__init__(cfg, server_args, _processor, *args, **kwargs)
        self.model_type = "qwen3_vl"

        self.fusion_audio_pad_id = int(getattr(hf_config, "audio_pad_id", 151654))
        self.fusion_n_query = int(getattr(hf_config, "n_query", 64))
        self.fusion_audio_model = getattr(hf_config, "audio_model", DEFAULT_AUDIO_MODEL)
        self.audio_token_id = self.fusion_audio_pad_id

        # Rebuild the special-token table with the audio slot marker attached.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=cfg.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=cfg.video_token_id,
            audio_token=AUDIO_PLACEHOLDER,
            audio_token_id=self.fusion_audio_pad_id,
        ).build(_processor)

    # ------------------------------------------------------------- audio load
    @classmethod
    def _load_single_item(cls, data, modality, frame_count_limit=None,
                          audio_sample_rate: Optional[int] = None,
                          discard_alpha_channel: bool = True):
        # Audio must be decoded/resampled with the reference pipeline (soxr);
        # SGLang's default loader (torchcodec/torchaudio) resamples differently
        # and costs real clips measurable cosine vs the reference embedder.
        if modality == Modality.AUDIO and not cls._is_preprocessed_input(data):
            return _decode_audio_reference(data, audio_sample_rate or WHISPER_SR)
        return super()._load_single_item(
            data, modality, frame_count_limit, audio_sample_rate,
            discard_alpha_channel,
        )

    # ------------------------------------------------------- audio featurize
    def process_mm_data(self, input_text, images=None, videos=None, audios=None,
                        **kwargs):
        audios = audios or []
        if len(audios) > 1:
            raise ValueError(
                "FusionEmbeddingModel serves one audio clip per request "
                "(prompt '<|vision_pad|><|im_end|>'); send one request per clip"
            )
        # The base HF processor (Qwen3VL) knows nothing about audio; featurize here.
        ret = super().process_mm_data(input_text, images=images, videos=videos,
                                      **kwargs)
        if audios:
            fe = _get_audio_feature_extractor(self.fusion_audio_model)
            # Reference-exact featurization (modeling_fusion_embedding.embed_audio):
            # 30 s mel window, padded to max_length, then trimmed by attention mask
            # (the trim happens model-side from audio_feature_lens).
            feats = fe(
                [np.asarray(a, dtype=np.float32) for a in audios],
                sampling_rate=fe.sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
            )
            ret["input_features"] = feats["input_features"]
            ret["audio_feature_lens"] = feats["attention_mask"].sum(dim=-1)
            ret["input_ids"] = self._expand_audio_placeholders(ret["input_ids"])
        return ret

    def _expand_audio_placeholders(self, input_ids):
        """Expand each single ``<|vision_pad|>`` audio marker to ``n_query`` slots
        (the HF processor only expands image/video pads)."""
        import torch

        flat = input_ids.flatten().tolist()
        expanded: list[int] = []
        for tok in flat:
            if tok == self.fusion_audio_pad_id:
                expanded.extend([self.fusion_audio_pad_id] * self.fusion_n_query)
            else:
                expanded.append(tok)
        return torch.tensor(expanded, dtype=input_ids.dtype).unsqueeze(0)
