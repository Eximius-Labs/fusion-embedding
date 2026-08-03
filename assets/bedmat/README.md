# Bed pressure mat, answered in language

An open-vocabulary posture readout for a commercial hospital-style pressure mat. The mat
streams 64x32 pressure frames; recognition is a text query against the shared
fusion-embedding space, so there is no trained posture classifier and no fixed label set.

Held-out subjects (11, 12, 13), never seen in training. Top-1 carries one sample standard
deviation over independent runs; single-run rows are marked.

| Metric | Result | Chance |
|---|---:|---:|
| 17-position top-1 (open-vocabulary, windowed), mean of 5 | **0.957 $\pm$ 0.016** | 0.059 |
| 17-position top-3, mean of 5 | 0.993 $\pm$ 0.010 | 0.176 |
| 17-position top-1, single frame (one run) | 0.947 | 0.059 |
| 3-class (supine / right / left) via the 17-way ranking (one run) | 0.987 | 0.333 |
| 3-class scored against **unseen coarse phrases**, all 7 runs | 0.882 | 0.333 |

The last row is identical in every run, MAE-initialized and from-scratch alike, because its
errors are not noise: 98.7% of them are the two positions where a subject lies on their
back with one knee raised, which the model ranks as lying on that side. Excluding those two
positions, coarse accuracy is 0.987. Whether a supine subject with the right knee up should
answer to "a person lying on their right side" is a question about the label taxonomy, not
about the model.

That coarse row is the open-vocabulary property in one number: the model was trained
against 17 fine-grained posture phrases and then scored against three coarse phrases it
never saw ("a person lying on their right side"), with no retraining and no classifier.

Published supervised baselines on this dataset report 99.6% on the 3-class task
(ResNet-18, leave-one-subject-out) and 82.7% in the dataset paper. Those are closed-set
classifiers: they answer three fixed labels and cannot answer a new phrase. Our 3-class
number (0.987, three held-out subjects) is in that range while the model is also doing the
17-way task and accepting arbitrary queries.

## Artifacts

All three artifacts were rendered from one MAE-initialized run scoring 0.961, near the
five-run mean.

- `bedmat_hero.gif` — the mat's heatmap on the left, ranked text queries on the right,
  updating as the subject changes posture. Every frame and every score is real model
  output on real held-out mat data.
- `bedmat_gallery.png` — eight held-out frames with their top-3 ranked queries. Selection
  policy: for each posture shown, the frame with the largest margin between the correct
  query and its nearest competitor (stated in the caption).
- `bedmat_timeline.png` — a night on the mat: the model's per-frame posture answer against
  the recorded posture (99% frame agreement), with the answer to "when did they last
  move?". The night is a disclosed composition: real recordings from one held-out subject,
  sessions concatenated in a fixed order.

## Model

Same recipe as [Tactus](https://huggingface.co/EximiusLabs/fusion-embedding-2-tactus),
retrained for this sensor: a 16.2M-parameter head (13.5M conv trunk, 2.6M projector) into the frozen
2048-d text space, masked-autoencoder pretraining on unlabeled frames from the training
subjects only, then contrastive training against the 17 posture phrases. The language side
is never updated. Training the head end to end costs a few dollars and about half an hour
on one A10G, which is the point: a new pressure sensor does not need a new foundation
model, it needs an afternoon.

Masked-autoencoder initialization does **not** clearly help on this sensor: 0.957 $\pm$
0.016 over five MAE-initialized runs against 0.945 $\pm$ 0.007 over three from-scratch runs
(difference +1.2 points, Welch t = 1.4, p = 0.22, which is under five test windows). The
+6.6-point same-sensor pretraining gain reported for the glove sensor does not reproduce
here. The likely reason is headroom rather than a failed lever: this task saturates near
0.96 with 20k frames of a single mat, where the glove task sat far from its ceiling.

## Data and license

[PhysioNet Pressure Map Dataset](https://physionet.org/content/pmd/1.0.0/) (pmd 1.0.0),
experiment I: 13 subjects, 17 in-bed postures, ~20k frames at 1 Hz on a Vista Medical FSA
SoftFlex 2048 (32x64 sensels). Released under ODC-By v1.0; please cite Pouyan et al., "A
pressure map dataset for posture and subject analytics," IEEE BHI 2017, and PhysioNet.
Subjects 1-10 train, 11-13 held out for every number on this page.

Not a medical device. This is a research demonstration, not a diagnostic tool.

## Reproduce

```
python scripts/bedmat_ingest.py --src <pmd-1.0.0 dir> --out <cache>   # parse + verify
modal run scripts/bedmat_train.py --action mae                        # same-sensor MAE
modal run scripts/bedmat_train.py --action train --steps 4000 \
    --init-from /vol/bedmat/bedmat_mae.pt
python scripts/make_bedmat_demo.py --cache <cache> --dump <dump.npz> --out assets/bedmat
```
