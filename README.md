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
  <a href="docs/images/SGIP_Poster_Preview.png"><img alt="Poster Preview" src="https://img.shields.io/badge/Poster-Preview-75529b?style=flat-square"></a>
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
- **Code release:** Code and pretrained checkpoints will be released soon. Please ⭐ star or watch this repository for updates.

## 🗂️ Release Plan

- [ ] Inference code
- [ ] Pretrained checkpoints
- [ ] Training code
- [ ] Data preprocessing and evaluation instructions

## Poster

<p align="center">
  <a href="docs/images/SGIP_Poster_Preview.png">
    <img src="docs/images/SGIP_Poster_Preview.png" alt="SGIP NeurIPS 2026 poster presenting motivation, contributions, method, results, and ablation studies" width="100%">
  </a>
</p>

[View full-resolution poster](docs/images/SGIP_Poster_Preview.png).

## 📖 Citation

```bibtex
@inproceedings{soon2026sgip,
  title     = {Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction},
  author    = {Soon, Donghyeon and Yoon, Jiil and Kim, Hyeondong and Park, Daehee},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
