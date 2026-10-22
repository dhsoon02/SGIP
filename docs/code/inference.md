# Inference and evaluation

First follow [setup](setup.md) and prepare the [input tensors](data.md).
A compatible checkpoint is required; pretrained weights will be released
later. Run commands from the repository root.

## Predict future motion

```bash
python predict.py \
  --input data/observed_inputs.pt \
  --output outputs/predictions.pt \
  --checkpoint checkpoints/motion.pth \
  --smplx_path data/body_models \
  --vposer_path vposer_v1_0 \
  --batch_size 1
```

The output is a tensor dictionary:

- `poses_pred`: `[N, 10, 38]`, the predicted orientation, translation, and
  VPoser latent sequence.
- `joints_pred`: `[N, 10, 23, 3]`, the final GCN-refined joint sequence.

Only future frames are saved. Each sample is processed in input order,
including the final incomplete batch. The code loads no LLM or teacher during
motion inference. It requires the precomputed selected-region points described
in [data.md](data.md).

For example, create one observed-input sample from prepared caches without
reading any future labels:

```python
from pathlib import Path
import torch

cache = Path('data/GIMO/SLICES_888s/test')
files = {
    'poses_input': 'poses_input.pth',
    'joints_input': 'joints_input.pth',
    'scene_points': 'scene_points_32768.pth',
    'objects': 'student_object_points.pth',
    'object_mask': 'student_object_points_mask.pth',
}
inputs = {key: torch.load(cache / name, map_location='cpu', weights_only=True)[:1]
          for key, name in files.items()}
torch.save(inputs, 'data/observed_inputs.pt')
```

## Evaluate a checkpoint

```bash
python evaluate.py \
  --dataroot data/GIMO \
  --checkpoint checkpoints/motion.pth \
  --object_points_name student_object_points.pth \
  --smplx_path data/body_models \
  --vposer_path vposer_v1_0 \
  --output results/test_metrics.json
```

The output contains the number of evaluated samples and four errors:

| Field | Metric |
| --- | --- |
| `traj_path` | Mean future root-joint position error |
| `traj_dest` | Root-joint error at the final frame |
| `mpjpe_path` | Mean root-relative future joint-position error |
| `mpjpe_dest` | Root-relative joint-position error at the final frame |

Errors are averaged per sample over all test rows, without dropping the final
batch. They use the **same coordinate units as the input tensors**. Multiply
by 1,000 to obtain millimetres only when your caches are in metres. The metric
formulas match the four future-motion metrics in the upstream evaluator.

Use the model dimensions and preprocessing corresponding to your checkpoint.
The loader restores learned VPoser weights and skips SMPL-X's batch-dependent
state. Missing learned weights or unexpected keys produce an error instead of
evaluating a partially loaded model. Legacy experiment checkpoints may require
their own model variants and are not automatically converted.
