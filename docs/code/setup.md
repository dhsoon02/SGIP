# Setup

This initial release contains the motion model and inference/evaluation tools.
**Pretrained SGIP checkpoints are not available yet.** The final training
recipes and interaction-prior generation pipeline will be released after
consolidation. Running the commands below requires your own compatible
checkpoint and prepared inputs.

## Environment

The setup follows the public [GIMO](https://github.com/y-zheng18/GIMO#quickstart)
and [SIF3D](https://github.com/kjle6/SIF3D-master#setup) environments. Use a
separate environment with versions compatible with their legacy PointNet2 and
VPoser interfaces. An exact GPU environment lock will accompany the confirmed
checkpoint release.

Install a CUDA-enabled PyTorch build appropriate for your machine using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/dhsoon02/SGIP.git
cd SGIP
python -m pip install -r requirements.txt
```

## PointNet++

SGIP needs both the compiled `pointnet2_ops` extension and the `pointnet2`
Python package: its backbone inherits `PointNet2ClassificationSSG`.
Start from [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
(the [o2oafford setup](https://github.com/daerduoCarey/o2oafford/tree/main/exps)
is the reference linked by GIMO and SIF3D):

```bash
# Run from the SGIP repository root.
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch ../Pointnet2_PyTorch
cd ../Pointnet2_PyTorch
```

Before building, apply the two compatibility edits described by GIMO/SIF3D.
Locate the code by function rather than by the line numbers in older READMEs:

1. In
   [`sampling_gpu.cu`](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu),
   find `furthest_point_sampling_kernel` and disable the magnitude check that
   skips points near the origin:

   ```cpp
   // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
   // if (mag <= 1e-3) continue;
   ```

2. In
   [`pointnet2_modules.py`](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py),
   find `PointnetFPModule.forward`. In the `known is None` branch, replace
   the `known_feats.expand(...)` expression with:

   ```python
   interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])
   ```

   Keep the three-neighbor interpolation in the `known is not None` branch.

Then install the extension and Python package from that checkout:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
cd ../SGIP
```

These commands use PointNet2's own requirements, which include legacy
Lightning/Hydra versions. Installing only `pointnet2_ops` is insufficient.
Use a CUDA toolkit/compiler compatible with your PyTorch installation.
Check which module is imported before inference:

```bash
python -c "import pointnet2_ops.pointnet2_modules as m; print(m.__file__)"
python -c "from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG"
```

The imported module should contain the edits above. Changes to CUDA source
require rebuilding the extension. Full GPU compatibility of this installation
has not yet been validated against a released SGIP checkpoint.

## VPoser and SMPL-X

Install a compatible [human_body_prior](https://github.com/nghorbani/human_body_prior)
version exposing `human_body_prior.tools.model_loader.load_vposer` and the
[SMPL-X Python package](https://github.com/vchoutas/smplx) (included in our
`requirements.txt`). Obtain the complete **VPoser v1.0 snapshot** and
**neutral SMPL-X NPZ model** through the [SMPL-X project](https://smpl-x.is.tue.mpg.de/).
The model assets are separate downloads and are not bundled here.

The research VPoser snapshot includes a rotation-conversion compatibility
adjustment; use the snapshot matching your checkpoint. A newer VPoser model
with a different interface cannot be substituted directly.

```bash
python -c "from human_body_prior.tools.model_loader import load_vposer"
python -c "import smplx"
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
