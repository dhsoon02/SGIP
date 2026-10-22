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

## Code

This initial release provides the motion model and tools for inference and
evaluation on prepared inputs. **Checkpoints are not included yet.** The
selected-region point clouds must already be available; the final prior
generation and training pipeline will follow.

| Entry point | Purpose |
| --- | --- |
| [`predict.py`](predict.py) | Predict future motion from observed inputs |
| [`evaluate.py`](evaluate.py) | Evaluate a compatible checkpoint |
| [`models/`](models/) | Motion model, attention blocks, and backbones |
| [`datasets/`](datasets/) | Prepared GIMO-format cache loader |

Start with **[Setup](docs/code/setup.md)** → **[Input data](docs/code/data.md)**
→ **[Inference and evaluation](docs/code/inference.md)**.
See [release scope](docs/code/release.md) for included components and provenance.

```bash
python predict.py --help
python evaluate.py --help
```

## 📖 Citation

```bibtex
@inproceedings{soon2026sgip,
  title     = {Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction},
  author    = {Soon, Donghyeon and Yoon, Jiil and Kim, Hyeondong and Park, Daehee},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
