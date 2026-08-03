# fusion-embedding for FiftyOne

Language-searchable embeddings for the sensor streams FiftyOne already synchronizes.

FiftyOne ingests MCAP natively and plays camera, LiDAR, IMU and robot state back on one
timeline, then offers embeddings and semantic search over it. That search runs on
image-language models, so the non-camera streams it went to the trouble of synchronizing
are the ones you cannot query in words. This source closes that gap: pressure, thermal,
motion and audio go into the same 2048-d space as text, and the App's existing similarity
panel answers plain-English queries over them.

Verified end to end against FiftyOne with real held-out sensor data: three natural-language
queries each sorted the correct posture to the top of a 17-window dataset
(`scripts/fiftyone_plugin_smoke.py`).

## Install

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/Eximius-Labs/fusion-embedding")
```

```bash
pip install fusion-embedding
```

| Model | Reads |
|---|---|
| `eximius/fusion-embedding-2` | images, in the shared space with text, video and audio |
| `eximius/ember` | thermal infrared images |
| `eximius/tactus` | touch, 32x32 pressure or taxel arrays |
| `eximius/tactus-mat` | touch, 64x32 body pressure mats |

## Images and thermal

These read media directly, so the standard recipe works:

```python
import fiftyone.brain as fob

fob.compute_similarity(dataset, model="eximius/ember", brain_key="thermal")
dataset.sort_by_similarity("a person standing in a doorway", brain_key="thermal")
```

## Embedding sensor fields

FiftyOne's `Model.media_type` currently admits only `"image"` and `"video"`, so a pressure
array or an inertial window cannot be embedded by pointing a model at a sample's media. The
sensor data lives in a field instead, and the recipe is to embed that field yourself and
hand the vectors to `compute_similarity`:

```python
import numpy as np
import fiftyone.brain as fob
import fiftyone.zoo as foz

model = foz.load_zoo_model("eximius/tactus-mat")

embeddings = []
for sample in dataset.iter_samples(progress=True):
    window = np.load(sample["pressure_window"])      # [F, 64, 32]
    embeddings.append(model.embed_sensor(window))

fob.compute_similarity(
    dataset,
    embeddings=np.stack(embeddings),
    model="eximius/tactus-mat",        # name it as a STRING, see below
    brain_key="touch",
)

dataset.sort_by_similarity("a person lying on their left side", brain_key="touch")
```

**Name the model as a string, not an instance.** `compute_similarity` reads
`can_embed_prompts` off whatever you pass, but an index built from a `Model` instance cannot
re-load that model later, so text queries stop working in the App and in future sessions.
FiftyOne warns about this; passing the zoo name is what makes prompt search persist.

The plugin under `plugin/` does the same thing from the App, so a user can point at a field
and get a text-searchable index without writing code:

```bash
fiftyone plugins download \
    https://github.com/Eximius-Labs/fusion-embedding \
    --plugin-names @eximius/fusion-embedding
```

## Notes

- Prompts are embedded through the family's canonical whitened readout, which is the same
  readout each sense pack was trained against. Embedding text any other way, including with
  the plain base model, misranks.
- A different sensor geometry needs a short fine-tune rather than a new model. Our own
  cross-sensor ablation says pooling across sensor families does not transfer, so treat a
  new mat or glove as a fine-tune target, not a zero-shot one.
- Weights: the sense packs are CC-BY-NC-4.0, except `eximius/tactus-mat`, which is ODC-By
  1.0 and free for commercial use with attribution. Code here is Apache-2.0.
- Models, numbers and limitations: https://huggingface.co/EximiusLabs
