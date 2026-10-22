# Dataset and input preparation

The released model consumes **six observed frames** and predicts **ten future
frames** at **2 fps**. Its upstream GCN and observed-frame splice are fixed to
this protocol. All points and joints must use the same transformed coordinate
frame and units as the checkpoint's training data.

## Download the raw dataset

Obtain GIMO through its official [dataset access instructions](https://github.com/y-zheng18/GIMO#dataset).
Complete the access request described there and follow the dataset agreement.
This repository distributes code and documentation; download raw data directly
from the dataset authors. Processed dataset files are not bundled.

A typical raw-data layout is:

```text
GIMO/
├── dataset.csv
└── <scene>/
    ├── scene_obj/
    │   ├── scene_downsampled.ply
    │   └── transform_norm.txt
    └── <sequence>/
        ├── smplx_local/
        │   └── <frame>.pkl
        ├── <transformation file>.json
        ├── eye_pc/
        └── PV/
```

The transformation filename is specified by the `transformation` column of
`dataset.csv`; do not assume one filename for every sequence. See the
[GIMO data description](https://github.com/y-zheng18/GIMO/blob/main/demo_data/README.md)
and [coordinate alignment code](https://github.com/y-zheng18/GIMO/blob/main/dataset/eval_dataset.py)
for the complete raw format. The released SGIP motion model does not consume
the raw gaze or image folders.

## Preparation status

The public SGIP loader currently reads **prepared tensor caches**. It does not
run preprocessing on first launch. The
[SIF3D preprocessing implementation](https://github.com/kjle6/SIF3D-master/blob/master/dataset/gimo_dataset.py)
is a reference for motion/scene alignment, frame sampling, and SMPL-X joint
conversion. It does not generate SGIP's student-selected region points.

The complete SGIP preparation pipeline will be released after its final
configuration is confirmed. It must produce aligned motion/scene tensors and
selected-region points in the contracts below. Final benchmark reproduction
instructions will accompany that release.

## Motion-only inference

`predict.py` reads a `.pt` dictionary with these tensors:

| Key | Shape | Description |
| --- | --- | --- |
| `poses_input` | `[N, 6, 38]` | Axis-angle orientation (3), translation (3), VPoser latent (32) |
| `joints_input` | `[N, 6, J, 3]`, `J >= 23` | Observed SMPL-X joints; the first 23 are used |
| `scene_points` | `[N, 32768, 3]` | Scene point cloud; size controlled by `--sample_points` |
| `objects` | `[N, P, 3]` | Selected region's point cloud, normally `P = 1024` |
| `object_mask` | `[N]` | Boolean mask indicating a valid selected region |

All coordinates/latents must be finite floating-point tensors. Samples with a
false object mask have their object points zeroed. The model does not receive
future poses, gaze, intent text, or teacher outputs through this interface.

**The selected-region points must already be available.** This release does
not yet turn raw scenes into a learned interaction prior. For observation-only
student evaluation, use points selected from observed information. A teacher-
or ground-truth-selected object cache is a different evaluation setting.

## GIMO-format evaluation caches

`evaluate.py` reads the test split in this layout:

```text
data/GIMO/
  dataset.csv
  SLICES_888s/test/
    poses_input.pth                  # [N, 6, 38]
    poses_label.pth                  # [N, 10, 38]
    joints_input.pth                 # [N, 6, J, 3]
    joints_label.pth                 # [N, 10, J, 3]
    scene_points_32768.pth           # [N, 32768, 3]
    student_object_points.pth        # [N, 1024, 3]
    student_object_points_mask.pth   # [N]
```

`dataset.csv` must contain `training`, `sequence_path`, `scene`,
`transformation`, `start_frame`, and `end_frame`. Evaluation rows have
`training = 0`. Their order must exactly match the cached tensor rows.
The loader checks row counts and tensor dimensions.

Use `--object_points_name` to select a differently named point cache and
`--object_mask_name` to override its mask. Otherwise, `_mask.pth` is derived
from the point-cache filename. Paths may be absolute or relative to the split
directory; `{split}` is expanded to `test` during evaluation.

The historical `gt_object_point_prior_*.pth` files are optional and are not
passed into the released model by the evaluation command. Future joint
labels are used only to calculate errors. `poses_label.pth` is retained in the
loader's cache contract for compatibility with the upstream dataset.

## Using existing GIMO or SIF3D caches

Existing caches are a starting point only after checking their protocol:

| Check | SGIP requirement |
| --- | --- |
| Frame sampling | 6 observed + 10 future frames, sampled at 2 fps |
| Motion representation | 38 values per frame: orientation, translation, VPoser latent |
| Joint representation | SMPL-X ordering, at least the first 23 joints |
| Coordinate system | Motion, scene points, and selected-region points share the checkpoint's frame and units |
| Sample alignment | Every cache row matches the same filtered `dataset.csv` row |
| Object conditioning | Student-selected region points and a validity mask for each sample |

SIF3D's README shows `SLICES_8s`, while its public loader uses `SLICES_888s`.
This release uses **`SLICES_888s`**, matching `datasets/gimo.py`. Renaming a
folder alone does not establish compatibility; verify the contents above.
SGIP does not require the `gazes.pth` cache produced by SIF3D.

The released evaluator reads the test split (`training = 0`). Preserve the
original split and row order when preparing caches. Object selection for
observation-only evaluation must use observed information; future labels and
teacher-selected targets belong to different settings.

After preparing compatible caches, set `--dataroot` to the directory containing
`dataset.csv` and `SLICES_888s`. Follow the [evaluation example](inference.md#evaluate-a-checkpoint)
with your SGIP checkpoint. GIMO/SIF3D pretrained weights are not substitutes
for an SGIP checkpoint.

The final PROX preparation protocol will be documented with the confirmed
training and preprocessing pipeline.
