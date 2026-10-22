<div align="center">

<h1>SGIP: Learning Scene-Grounded Interaction Priors<br>for Scene-Aware Human Motion Prediction</h1>

<p><strong>NeurIPS 2026</strong></p>

<p>
  <a href="https://dhsoon02.github.io/"><strong>Donghyeon Soon</strong></a><sup>1*</sup> ·
  <a href="https://yoon-ji-il.github.io/"><strong>Jiil Yoon</strong></a><sup>1*</sup> ·
  <a href="https://www.hyeondong.kim/"><strong>Hyeondong Kim</strong></a><sup>1</sup> ·
  <a href="https://sites.google.com/view/dhpark/"><strong>Daehee Park</strong></a><sup>1,2†</sup><br>
  <sup>1</sup>DGIST &nbsp;·&nbsp; <sup>2</sup>UNIVA<br>
  <sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author
</p>

<p>
  <a href="docs/paper.pdf"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-2f80ed?style=flat-square"></a>
  <a href="https://sgip2026.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2ea44f?style=flat-square"></a>
  <a href="docs/poster.pdf"><img alt="Poster PDF" src="https://img.shields.io/badge/Poster-PDF-75529b?style=flat-square"></a>
  <a href="https://neurips.cc/virtual/2026/loc/sydney/poster/152697"><img alt="NeurIPS 2026" src="https://img.shields.io/badge/NeurIPS-2026-6843b5?style=flat-square"></a>
</p>

</div>

<p align="center">
  <a href="docs/images/paper_figures/teaser_camready.pdf">
    <img src="docs/images/paper_figures/teaser_camready.png" alt="SGIP teaser: previous gaze- and distance-based methods leave future grounding ambiguous; SGIP predicts an explicit interaction prior over objects and open space to ground future motion without extra sensing at inference" width="100%">
  </a>
</p>

**TL;DR** Given observed motion and a 3D scene, **SGIP** predicts an explicit interaction
prior over objects and open space to guide future trajectory and full-body pose
prediction. Intent distillation transfers training-time guidance from an intent-aware
teacher to an observation-only student, with **no gaze, textual intent, or LLM at inference**.

## 📢 News

- 🎉 SGIP is accepted to **NeurIPS 2026!** See you in Sydney!
- **Code release:** Core motion inference and evaluation code is now available. Pretrained checkpoints, final training recipes, and the interaction-prior pipeline will be released after cleanup. Please ⭐ star or watch this repository for updates.

## 🗂️ Release Plan

- [x] Core motion inference and evaluation code
- [x] Setup and input/output documentation
- [ ] Pretrained checkpoints
- [ ] Final teacher/student training code and configurations
- [ ] Interaction-prior generation and data preprocessing

## Setup

Our environment and data conventions build on
[GIMO](https://github.com/y-zheng18/GIMO) and
[SIF3D](https://github.com/kjle6/SIF3D-master).

Install a CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) build
for your machine, then install the Python dependencies:

```bash
git clone https://github.com/dhsoon02/SGIP.git
cd SGIP
python -m pip install -r requirements.txt
```

Install **PointNet++**, **VPoser**, and **SMPL-X** following our
[setup guide](docs/code/setup.md). It includes the PointNet++ compatibility
edits used by GIMO/SIF3D and the required body-model assets.

## Dataset

### Download GIMO

Request the raw dataset through the official
[GIMO download instructions](https://github.com/y-zheng18/GIMO#dataset).
Please follow the dataset's usage agreement. Raw data and processed dataset
files are not distributed in this repository.

The download contains `dataset.csv`, scene scans, and per-sequence motion
files. See the [official data description](https://github.com/y-zheng18/GIMO/blob/main/demo_data/README.md)
for their organization and our [data guide](docs/code/data.md) for the files
used by SGIP.

### Prepare SGIP inputs

The released motion model uses **6 observed frames and 10 future frames at
2 fps**. Evaluation expects the following prepared caches:

```text
data/GIMO/
├── dataset.csv
└── SLICES_888s/test/
    ├── poses_input.pth
    ├── poses_label.pth
    ├── joints_input.pth
    ├── joints_label.pth
    ├── scene_points_32768.pth
    ├── student_object_points.pth
    └── student_object_points_mask.pth
```

SGIP additionally needs the **student-selected region points and validity
mask**. Gaze files are not used by the released motion model. Motion, scene,
and region caches must have matching sample order and coordinates.

**The current loader reads prepared caches.** Automatic raw-data preprocessing
and interaction-prior generation will be released with the confirmed pipeline.
The [data guide](docs/code/data.md#using-existing-gimo-or-sif3d-caches) explains
what to check when starting from GIMO/SIF3D data.

## Quickstart

**Pretrained SGIP checkpoints are coming later.** The examples below require
a compatible SGIP checkpoint, body-model assets, and prepared inputs. Checkpoint
paths are placeholders for your local files.

### Evaluation

```bash
python evaluate.py \
  --dataroot /path/to/GIMO \
  --checkpoint /path/to/sgip_motion.pth \
  --smplx_path /path/to/body_models \
  --vposer_path /path/to/vposer_v1_0 \
  --output results/gimo_metrics.json
```

### Inference

Create an observed-input tensor bundle as described in the
[inference guide](docs/code/inference.md), then run:

```bash
python predict.py \
  --input data/observed_inputs.pt \
  --output outputs/predictions.pt \
  --checkpoint /path/to/sgip_motion.pth \
  --smplx_path /path/to/body_models \
  --vposer_path /path/to/vposer_v1_0
```

Training scripts, final configurations, and interaction-prior generation will
follow. See the [release scope](docs/code/release.md) for the current coverage.

| Code | Purpose |
| --- | --- |
| [`predict.py`](predict.py) / [`evaluate.py`](evaluate.py) | Inference and evaluation |
| [`models/`](models/) | Motion model, attention blocks, and backbones |
| [`datasets/`](datasets/) | Prepared GIMO-format cache loader |

## Acknowledgements

We thank the authors of [GIMO](https://github.com/y-zheng18/GIMO),
[SIF3D](https://github.com/kjle6/SIF3D-master), and
[PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch) for their public
implementations and documentation. See [third-party notices](THIRD_PARTY_NOTICES.md)
for additional dependencies.

## 📖 Citation

```bibtex
@inproceedings{soon2026sgip,
  title     = {Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction},
  author    = {Soon, Donghyeon and Yoon, Jiil and Kim, Hyeondong and Park, Daehee},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
