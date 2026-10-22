# Input data

The released model consumes **six observed frames** and predicts **ten future
frames** at **2 fps**. Its upstream GCN and observed-frame splice are fixed to
this protocol. All points and joints must use the same transformed coordinate
frame and units as the checkpoint's training data.

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

Datasets, prepared caches, raw-data preprocessing, and the final PROX protocol
are not included in this initial release. These will be documented with the
confirmed training and preprocessing pipeline.
