# fusion-embedding vLLM plugin

Out-of-tree [vLLM model plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html)
that serves the EximiusLabs **fusion-embedding-2** family
(`architectures: ["FusionEmbeddingModel"]`) through vLLM's pooling runner. Ships inside
the `fusion-embedding` package (module `fusion_embedding.vllm_plugin`, entry point group
`vllm.general_plugins`); it is inert unless a vLLM process loads it.

Developed and smoke-tested against **`vllm==0.26.0`**.

## What it does

- Registers the `FusionEmbeddingModel` architecture with vLLM (entry point group
  `vllm.general_plugins`, so a plain `vllm serve` picks it up — no code changes).
- Registers the repo's `fusion-embedding-connector` config type with transformers and
  merges it with the frozen base's config at parse time, so vLLM sees a full
  Qwen3VL-shaped config (no `trust_remote_code` needed).
- Loads the **frozen base** weights (`Qwen/Qwen3-VL-Embedding-2B`) from the base repo,
  the trained connector / whitening / gated adapters from the served fusion repo, and
  the **frozen audio tower** (`Qwen/Qwen2.5-Omni-7B`, `thinker.audio_tower.*` only)
  from the Omni repo.
- Text / image / video run the frozen base path (adapters are token-gated no-ops on
  non-audio sequences — the family's bitwise-identity guarantee). Text embeddings are
  whitened; non-text are not, exactly as in the reference readout.
- Audio is a first-class vLLM multimodal input (see below), executed as
  mel -> Omni audio tower -> perceiver-resampler -> 64 injected tokens, with the
  rank-384 adapters active only on the audio sequence's tokens.
- Output contract: last-token pooling, (text-only) whitening, L2-normalized, full
  2048-d by default. The MRL ladder is advertised as `matryoshka_dimensions`, so the
  `dimensions` parameter of the embeddings API selects shorter rungs (1024, 512, ...).
- Pooling position reproduces the released readout exactly. The base tokenizer
  appends `<|endoftext|>` under default tokenization; the reference pools TEXT at the
  assistant-opener newline (`add_special_tokens=False`) but pools IMAGES at the
  appended `<|endoftext|>` (HF processor default). The pooler therefore steps back
  over a trailing `<|endoftext|>` for non-vision sequences (text, audio) and keeps
  the true last token for vision sequences — causal attention makes the stepped-back
  hidden state bit-identical to a no-pad prompt.
- Audio resampling uses soxr (same library behind librosa's default `soxr_hq` used
  by the reference), not vLLM's pyav default.

## Install

```bash
pip install 'vllm[audio]==0.26.0' fusion-embedding
```

(The `[audio]` extra is required for the audio input path.)

## Serve

```bash
vllm serve EximiusLabs/fusion-embedding-2-2b-preview --runner pooling
```

The tokenizer/processor resolve from the base repo automatically (the fusion repo does
not carry tokenizer files). The plugin forces eager execution: the per-token adapter
gate is a Python-level hook and must never be captured into a compiled graph.
Single-GPU only (the trained modules are not tensor-parallelized).

Prompt formats (the model is instruction-formatted; send the full template):

- text query:
  `<|im_start|>system\nRetrieve images or text relevant to the user's query.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`
- image document (with an image attached):
  `<|im_start|>system\nRepresent the user's input.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n`
- audio (with an audio attached): `<|vision_pad|><|im_end|>` — one `<|vision_pad|>`
  per clip (the released checkpoint reuses this inert base token as its audio slot;
  the plugin expands it to the 64 audio positions) followed by `<|im_end|>`, matching
  the reference readout `[audio_pad]*64 + [eos]`.

## Offline API

```python
from vllm import LLM

llm = LLM(model="EximiusLabs/fusion-embedding-2-2b-preview", runner="pooling")

text_out = llm.embed("<|im_start|>system\nRetrieve images or text relevant to the "
                     "user's query.<|im_end|>\n<|im_start|>user\na dog barks<|im_end|>\n"
                     "<|im_start|>assistant\n")

import soundfile as sf
wav, sr = sf.read("dog.wav", dtype="float32")
audio_out = llm.embed({"prompt": "<|vision_pad|><|im_end|>",
                       "multi_modal_data": {"audio": (wav, sr)}})
```

## Environment switches (debug / staging only)

- `FUSION_VLLM_DISABLE_AUDIO=1` — skip the Omni tower entirely (text/image/video only;
  much lighter, no 7B snapshot download).
- `FUSION_VLLM_DISABLE_ADAPTERS=1` — frozen-base only (refuses to combine with the
  audio path, which requires the adapters).

## Parity (measured 2026-08-04, vllm==0.26.0, transformers 5.14.1)

`scripts/vllm_plugin_smoke.py` (Modal) checks the served embeddings against the
reference `fusion_embedding.UnifiedEmbedder` on identical inputs:

```bash
PYTHONUTF8=1 uv run --env-file .env modal run scripts/vllm_plugin_smoke.py --stage a
```

Both sides at fp32 (isolates implementation correctness from bf16 kernel rounding):

- Stage A (frozen base): text cosine 0.999998-0.999999 (4 prompts), image
  0.999986-0.999990 (2 images) — PASS at > 0.999.
- Stage B (adapters loaded, gates closed): every text/image vector identical to the
  adapter-free run with max_abs_diff exactly 0.0 (at fp32 AND bf16) — the token
  gate is a true no-op on non-audio sequences. One condition: the comparison pins
  the KV-cache allocation (`num_gpu_blocks_override`). Without the pin, loading the
  adapters shrinks free memory, the engine sizes the KV pool differently, and
  fused-attention kernel plans can change — a deterministic ~1e-4 drift that is
  engine kernel selection, not adapter math. Any bitwise A/B comparison across
  model configurations should pin the KV allocation on both sides.
- Stage C (full model, audio): cosine 0.999999-1.000000 on 6 clips (3 synthetic +
  3 real FreeSound wavs, incl. 44.1 kHz clips exercising resampling).

At the served precision (bf16 both sides): text 0.9993-0.9996, audio 0.9995-0.9999,
image 0.9979-0.9987. The image residual is bf16 kernel-rounding drift between
vLLM's and HF's vision/decoder kernels (any two bf16 implementations differ at this
level; the fp32 numbers above bound the implementation error), and the text-side
whitening amplifies pooled-vector rounding, which is why the fp32 gate exists.

`--stage serve` boots the literal `vllm serve ... --runner pooling` command and
checks HTTP /v1/embeddings against the bf16 reference: 0.9993-0.9996 — PASS.
