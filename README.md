<div align="center">

<h1>[NeurIPS 2026] Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction</h1>

<p>
  <a href="https://dhsoon02.github.io/"><strong>Donghyeon Soon</strong></a><sup>1*</sup> ·
  <a href="mailto:wldlf2819@dgist.ac.kr"><strong>Jiil Yoon</strong></a><sup>1*</sup> ·
  <a href="mailto:hdkim@dgist.ac.kr"><strong>Hyeondong Kim</strong></a><sup>1</sup> ·
  <a href="https://sites.google.com/view/dhpark/"><strong>Daehee Park</strong></a><sup>1,2†</sup><br>
  <sup>1</sup>DGIST &nbsp;·&nbsp; <sup>2</sup>UNIVA<br>
  <sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author
</p>

<p>
  <a href="docs/paper.pdf"><img alt="Paper" src="https://img.shields.io/badge/Paper-PDF-2f80ed?style=flat-square&logo=adobeacrobatreader&logoColor=white"></a>
  <a href="https://sgip2026.github.io/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-1b427d?style=flat-square&logo=googlechrome&logoColor=white"></a>
</p>

</div>

## News!

- 🎉 Our paper, **“Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction,”** has been accepted to **NeurIPS 2026!**

## Teaser

<p align="center">
  <a href="docs/images/paper_figures/h_teaser.pdf">
    <img src="docs/images/paper_figures/h_teaser.png" alt="SGIP teaser: ambiguous targets with gaze- and distance-based methods, an explicit interaction prior over objects and free space, and a predicted motion toward the sofa" width="100%">
  </a>
</p>

The same observed motion can lead to different objects or through open space,
making the future interaction target ambiguous. **SGIP** predicts an **explicit
interaction prior** over these scene regions and uses the selected target—here,
the sofa—to guide future trajectory and full-body pose prediction, using only
**observed motion and a 3D scene at inference**.

## Overview

<p align="center">
  <a href="docs/images/paper_figures/fig2.png">
    <img src="docs/images/paper_figures/fig2.png" alt="Overall framework of SGIP: intent-guided teacher, observation-only student, and scene-grounded motion prediction" width="100%">
  </a>
</p>

**SGIP** explicitly predicts where future human motion will be grounded in a 3D scene,
then uses the selected scene token to condition future trajectory and full-body pose
prediction. Its scene-grounded interaction prior covers both object instances and
open navigation space.

Through **intent distillation**, a training-only teacher combines LLM-based intent
reasoning with geometric motion-scene features. An observation-only student learns
from the teacher through prior-level and motion-level distillation. At inference,
SGIP requires only **observed motion and a 3D scene**, without gaze, textual intent,
or an LLM.

## Benchmarks

We evaluate SGIP on **GIMO** and **PROX** for future trajectory and full-body pose
prediction. SGIP outperforms the compared scene-aware baselines on both datasets
in the observation-only setting. Full quantitative and qualitative results are
available in the [paper](docs/paper.pdf) and on the
[project page](https://sgip2026.github.io/).

## Code

*Code will be released soon.*

The project website is maintained in [SGIP2026/sgip2026.github.io](https://github.com/SGIP2026/sgip2026.github.io).

## Citation

```bibtex
@inproceedings{soon2026sgip,
  title     = {Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction},
  author    = {Soon, Donghyeon and Yoon, Jiil and Kim, Hyeondong and Park, Daehee},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
