# Setup

This initial release contains the motion model and inference/evaluation tools.
**Pretrained SGIP checkpoints are not available yet.** The final training
recipes and interaction-prior generation pipeline will be released after
consolidation. Running the commands below requires your own compatible
checkpoint and prepared inputs.

## Environment

Use Python 3.10 or newer in a separate environment. Install a CUDA-enabled
PyTorch build appropriate for your machine using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/dhsoon02/SGIP.git
cd SGIP
python -m pip install -r requirements.txt
```

Two external dependencies require additional setup:

1. **PointNet2:** install both `pointnet2_ops` and the `pointnet2` Python
   package from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).
   The backbone inherits `PointNet2ClassificationSSG` and uses compiled CUDA
   operations. A CPU-only PyTorch installation cannot run the full model.
2. **VPoser and SMPL-X:** install a compatible
   [human_body_prior](https://github.com/nghorbani/human_body_prior) version
   that exposes `human_body_prior.tools.model_loader.load_vposer`. Obtain the
   complete **VPoser v1.0 snapshot** and **neutral SMPL-X NPZ model** from the
   [SMPL-X project](https://smpl-x.is.tue.mpg.de/). These assets are not included.

The upstream code uses legacy PointNet2/VPoser interfaces. The requirements
file lists direct dependencies; it is not a verified full GPU environment
lock. Use the dependency versions matching your checkpoint. In particular,
the upstream research VPoser snapshot includes a rotation-conversion
compatibility adjustment; an arbitrary newer VPoser model is not a substitute.

Check the installed interfaces before inference:

```bash
python -c "import pointnet2_ops; from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG"
python -c "from human_body_prior.tools.model_loader import load_vposer"
```

## External assets

The default layout is:

```text
SGIP/
  checkpoints/motion.pth                 # compatible SGIP state_dict
  data/body_models/smplx/SMPLX_NEUTRAL.npz
  vposer_v1_0/                          # full upstream snapshot, not only weights
  data/GIMO/                           # prepared caches; see data.md
```

Override locations with `--checkpoint`, `--smplx_path`, `--vposer_path`, and
`--dataroot`. `SMPLX_PATH`, `VPOSER_PATH`, and `DATAROOT` also set the defaults.
The command-line help works without loading CUDA extensions or body models:

```bash
python predict.py --help
python evaluate.py --help
```

## Lightweight checks

With PyTorch and the development requirements installed:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

These CPU checks cover input contracts, batching, metrics, data alignment,
and checkpoint loading. They do not reproduce the paper's reported scores.
See [release scope and provenance](release.md).
