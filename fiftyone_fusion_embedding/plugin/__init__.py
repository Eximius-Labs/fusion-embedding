"""FiftyOne plugin: embed a sensor field into the fusion-embedding shared space.

FiftyOne's ``Model.media_type`` currently admits only "image" and "video", so a pressure
array or an inertial window cannot be embedded by pointing a model at the sample's media.
This operator closes that gap: it reads a *field* on each sample, embeds it with a sense
pack, writes the vectors back, and builds a similarity index that still answers text
queries because the index names a zoo model for prompt support.

The field may hold either an inline array or a path to a ``.npy`` file, which is the shape
sensor data usually arrives in.
"""
import numpy as np

import fiftyone.operators as foo
import fiftyone.operators.types as types

SENSOR_MODELS = ("eximius/tactus-mat", "eximius/tactus")


def _load_window(value):
    """A field value -> a numpy sensor window."""
    if isinstance(value, str):
        return np.load(value)
    return np.asarray(value)


class ComputeFusionEmbeddings(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_fusion_embeddings",
            label="Fusion Embedding: embed a sensor field",
            description=(
                "Embed a pressure or sensor field into the shared text space and build a "
                "text-searchable similarity index"),
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        model_choices = types.Dropdown(label="Sense pack")
        for name in SENSOR_MODELS:
            model_choices.add_choice(name, label=name)
        inputs.enum(
            "model_name", model_choices.values(), view=model_choices, required=True,
            default=SENSOR_MODELS[0],
            description="Which sense pack reads this field")

        field_choices = types.Dropdown(label="Sensor field")
        schema = ctx.dataset.get_field_schema() if ctx.dataset is not None else {}
        for fname in schema:
            field_choices.add_choice(fname, label=fname)
        inputs.enum(
            "sensor_field", field_choices.values(), view=field_choices, required=True,
            description="Field holding the sensor window, inline or a path to a .npy")

        inputs.str(
            "embeddings_field", default="fusion_embedding", required=True,
            label="Write embeddings to",
            description="Vector field to create on each sample")
        inputs.str(
            "brain_key", default="fusion_sim", required=True,
            label="Brain key",
            description="Similarity index name; query it from the App's search bar")
        return types.Property(inputs, view=types.View(label="Fusion Embedding"))

    def execute(self, ctx):
        import fiftyone.brain as fob
        import fiftyone.zoo as foz

        model_name = ctx.params["model_name"]
        sensor_field = ctx.params["sensor_field"]
        emb_field = ctx.params["embeddings_field"]
        brain_key = ctx.params["brain_key"]

        model = foz.load_zoo_model(model_name)

        view = ctx.dataset
        embeddings, ids, skipped = [], [], 0
        for sample in view.iter_samples(progress=True):
            value = sample[sensor_field]
            if value is None:
                skipped += 1
                continue
            vec = model.embed_sensor(_load_window(value))
            sample[emb_field] = vec.tolist()
            sample.save()
            embeddings.append(vec)
            ids.append(sample.id)

        if not embeddings:
            raise ValueError(
                f"no samples had a value in '{sensor_field}'; nothing to embed")

        # Naming the model as a STRING is what preserves prompt support in the App and in
        # later sessions; passing an instance would build an index that cannot answer text.
        fob.compute_similarity(
            view.select(ids),
            embeddings=np.stack(embeddings),
            model=model_name,
            brain_key=brain_key,
        )
        return {
            "embedded": len(embeddings),
            "skipped": skipped,
            "brain_key": brain_key,
            "embeddings_field": emb_field,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.int("embedded", label="Samples embedded")
        outputs.int("skipped", label="Samples skipped (empty field)")
        outputs.str("brain_key", label="Similarity index")
        outputs.str("embeddings_field", label="Embeddings field")
        return types.Property(outputs, view=types.View(label="Done"))


def register(p):
    p.register(ComputeFusionEmbeddings)
