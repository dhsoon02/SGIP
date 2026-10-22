"""Future-only trajectory and root-relative joint errors in input units."""

import torch


def motion_errors(prediction, observed, future):
    """Return one value per sample, matching the upstream evaluation metrics."""
    if prediction.ndim != 4 or observed.ndim != 4 or future.ndim != 4:
        raise ValueError('Expected [batch, time, joints, 3] tensors.')
    expected = torch.cat([observed, future], dim=1)
    if prediction.shape != expected.shape or prediction.shape[-1] != 3:
        raise ValueError(f'Prediction/target shape mismatch: {prediction.shape} vs {expected.shape}')
    prediction = prediction[:, observed.shape[1]:]
    target = future
    trajectory = torch.linalg.vector_norm(prediction[:, :, 0] - target[:, :, 0], dim=-1)
    pred_relative = prediction - prediction[:, :, :1]
    target_relative = target - target[:, :, :1]
    pose = torch.linalg.vector_norm(pred_relative - target_relative, dim=-1)
    return {
        'traj_path': trajectory.mean(dim=1),
        'traj_dest': trajectory[:, -1],
        'mpjpe_path': pose.mean(dim=(1, 2)),
        'mpjpe_dest': pose[:, -1].mean(dim=1),
    }
