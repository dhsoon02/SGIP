# SGIP · NeurIPS 2026

**Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction**

Donghyeon Soon\*, Jiil Yoon\*, Hyeondong Kim, Daehee Park† — DGIST<br>
\* Equal contribution. † Corresponding author.

SGIP learns an explicit distribution over scene objects and navigable space to
condition future human trajectories and full-body poses. Privileged intent
supervises a teacher during training; the deployed student requires only
observed motion and a 3D scene, without gaze, textual intent, or an LLM.

- [Paper PDF](docs/paper.pdf)
- [Research overview video](docs/assets/videos/sgip-overview.mp4) — 72 seconds, 1080p, English on-screen explanations, no audio
- [BibTeX](docs/citation.bib)
- Project page address after GitHub Pages deployment: <https://dhsoon02.github.io/SGIP/>

This repository currently contains the research project page and its media.

## Preview locally

No JavaScript build step or npm dependencies are required.

```bash
python scripts/preview.py
```

Open <http://localhost:8000>. The preview server supports HTTP range requests
for video seeking. It binds to localhost; use `--port 8080` to change the port.

## Publish on GitHub Pages

Push the repository changes, then choose **Settings → Pages → Deploy from a
branch → main → /docs → Save**. All assets use relative paths and work under
the `/SGIP/` project path. `docs/.nojekyll` bypasses Jekyll processing.

## Files

```text
docs/
  index.html                 Research project page
  .nojekyll                  Static GitHub Pages publishing
  paper.pdf                  Paper and appendix compiled from the supplied LaTeX
  citation.bib               Citation download
  assets/
    css/style.css            Responsive page styles
    js/main.js               Accessible tabs, figure viewer, and citation copy
    fonts/                   Self-hosted Manrope and its OFL license
    images/                  Paper figures, panel crops, and posters
    videos/                  H.264 overview and English WebVTT captions
scripts/
  preview.py                 Local static server with seekable video playback
  prepare_figures.py         Render web figures from the manuscript ZIP
  build_paper.py             Compile the manuscript for the Paper link
  render_overview.py         Reproduce the overview from the manuscript ZIP
  requirements-media.txt     Python dependencies for media generation
```

## Regenerate assets

Install the media dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements-media.txt
python scripts/prepare_figures.py '/path/to/manuscript.zip'
```

To regenerate the video, install FFmpeg with `libx264` and DejaVu Sans fonts:

```bash
python scripts/render_overview.py --paper '/path/to/manuscript.zip' \
  --output docs/assets/videos/sgip-overview.mp4
```

Copy the generated `sgip-overview-poster.jpg` to
`docs/assets/images/video-poster.jpg` if the title slide changes. The generator
also writes English captions and chapter stills. Use `--stills-only` to review
layouts without encoding a video.

The overview uses original paper figures with editorial transitions, text
reveals, and highlights. It is an illustrated research explanation, not a
playback of generated or reconstructed human-motion sequences. Real motion
videos can be added later from model outputs.

To regenerate the PDF, install [Tectonic](https://tectonic-typesetting.github.io/):

```bash
python scripts/build_paper.py '/path/to/manuscript.zip'
```

The paper build works on a temporary copy. It omits the literal placeholder
acknowledgments block (`Acknowledgments Part!!`) and converts Unicode en/em
dashes to equivalent TeX notation for compatibility with the manuscript font.
The original archive, research claims, results, and authorship are unchanged.

## Content notes

- Benchmark values come from `bodies/4_experiments.tex` in the supplied manuscript.
- The 14.5% GIMO Traj-path reduction is `(642 − 549) / 642` relative to gaze-free GAP3DS*.
- The 12.6% PROX Traj-dest reduction is `(659 − 576) / 659` relative to gaze-free SIF3D*.
- Observation-only models are shown separately from models using additional or privileged cues.
- Main benchmark results use hard top-1 grounding-token selection; alternative-future figures illustrate token-conditioned alternatives.
- No arXiv identifier, proceedings volume, or page numbers have been invented.

The independently implemented layout is inspired by
[LightSplat](https://vision3d-lab.github.io/lightsplat/) and
[Nerfies](https://nerfies.github.io/). Manrope uses the SIL Open Font License;
see `docs/assets/fonts/OFL.txt`.

## Citation

```bibtex
@inproceedings{soon2026sgip,
  title     = {Learning Scene-Grounded Interaction Priors for Scene-Aware Human Motion Prediction},
  author    = {Soon, Donghyeon and Yoon, Jiil and Kim, Hyeondong and Park, Daehee},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
