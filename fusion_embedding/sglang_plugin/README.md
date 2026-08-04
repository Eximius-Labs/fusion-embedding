# fusion-embedding SGLang plugin

Out-of-tree [SGLang](https://github.com/sgl-project/sglang) model plugin that serves
[EximiusLabs/fusion-embedding-2-2b-preview](https://huggingface.co/EximiusLabs/fusion-embedding-2-2b-preview)
as an embedding model, with parity against the released reference embedder.

## Install and launch

Developed and gated against **sglang==0.5.16** (pin it; the plugin relies on the
`SGLANG_EXTERNAL_*` hooks and the in-tree pooling Qwen3-VL model of that version).

```bash
pip install sglang==0.5.16
pip install 'fusion-embedding[sense]'

SGLANG_EXTERNAL_MODEL_PACKAGE=fusion_embedding.sglang_plugin.models \
SGLANG_EXTERNAL_MM_MODEL_ARCH=FusionEmbeddingModel \
SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=fusion_embedding.sglang_plugin.processors \
python -m sglang.launch_server \
    --model-path EximiusLabs/fusion-embedding-2-2b-preview --is-embedding
```

The package also registers an `sglang.srt.plugins` entry point that defaults the three
env vars, registers the repo's shell config on `transformers.AutoConfig` (no
`trust_remote_code`), and applies the model-specific launch adjustments — so once the
package is pip-installed, the env exports are optional.

Automatic launch adjustments (applied only when the model resolves to the fusion
connector):

- `tokenizer_path` -> `Qwen/Qwen3-VL-Embedding-2B` (the fusion repo carries no
  tokenizer/processor files),
- CUDA graphs and `torch.compile` off — the rank-384 token-gated adapters run as
  Python forward hooks and must never be captured or compiled,
- radix prefix cache off and chunked prefill off — the readout contract pools with a
  modality-aware step-back over a trailing pad, which requires whole sequences in one
  extend and no cross-request prefix reuse,
- refuses to launch without `--is-embedding`.

## What the model is

- **Frozen base**: `Qwen/Qwen3-VL-Embedding-2B` (weights pulled from the base repo at
  load time). Text/image/video run bit-for-bit the frozen base path.
- **Trained parts** (the fusion repo's `model.safetensors`): perceiver-resampler
  audio connector (64 latent tokens), diagonal text whitening, and modality-gated
  bottleneck adapters (rank 384, all 28 decoder layers) that execute only on tokens
  of audio requests.
- **Frozen audio tower**: `Qwen/Qwen2.5-Omni-7B` `thinker.audio_tower` (pulled from
  its repo at load time).

## Request formats

Embeddings come from the OpenAI-compatible `/v1/embeddings` route, the native
`/encode` route, or `sglang.Engine(..., is_embedding=True).encode(...)`.

Text (query):

```
<|im_start|>system
Retrieve images or text relevant to the user's query.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
```

Image (document), with the image attached as `image_data`:

```
<|im_start|>system
Represent the user's input.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|><|im_end|>
<|im_start|>assistant
```

Audio: prompt `<|vision_pad|><|im_end|>` with the clip attached as `audio_data`
(path, URL, base64 `data:` URL, or raw bytes). One clip per request. The processor
decodes with soundfile and resamples with librosa's `soxr_hq` — the resampler the
released checkpoint was trained with (SGLang's default torchaudio/torchcodec loader
drifts audibly from the reference).

Matryoshka truncation is served through the standard `dimensions` parameter
(2048/1536/1024/512/256/128/64).

## Readout contract notes (deliberate, do not "fix")

- TEXT pools at the assistant-opener newline; the server-default tokenization appends
  `<|endoftext|>`, so the pooler steps back over exactly one trailing pad for
  non-vision sequences (exact under causal attention).
- IMAGES pool AT the appended `<|endoftext|>` (HF processor default) — the asymmetry
  is part of the released contract.
- Whitening applies to pure-text requests only, on the raw pooled vector, before MRL
  truncation and L2 normalization.

## Parity harness notes (fp32)

The structural parity gates run both sides at fp32. SGLang's fused sgl-kernel ops
(rmsnorm, silu_and_mul) and its fused attention backends dispatch half dtypes only,
so fp32 serving needs:

- `--attention-backend torch_native --mm-attention-backend sdpa`,
- `FUSION_SGLANG_NATIVE_OPS=1` — the plugin swaps RMSNorm/SiluAndMul for their
  native torch implementations before the model is built (set automatically by the
  ServerArgs hook when `--dtype float32`).

When comparing configurations at bf16 (e.g. adapters loaded vs frozen base), pin
`--max-total-tokens`: a memory-derived KV pool size changes the fused attention
kernel plans between configurations, a deterministic engine-level drift (~3e-3)
unrelated to the model. With the pool pinned, adapters-loaded vs frozen-base is
bitwise identical on text and image at both fp32 and bf16.

## Known limitations

- One audio clip per request (the released prompt format); batch clips as separate
  requests.
- Video via this plugin is untested (the reference video path was not part of the
  parity gates).
- Tensor parallel / pipeline parallel untested; serve with the default single-GPU
  configuration.
- `sglang==0.5.16` pins `transformers==5.12.1`; the plugin relies on the in-tree
  `qwen3_vl` and `qwen2_5_omni` model code of that version.
