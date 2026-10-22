"""Shared checkpoint and body-model setup for inference and evaluation."""

from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_motion_weights(model, path, device):
    """Restore learned parameters while keeping batch-dependent SMPL-X buffers."""
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a plain motion-model state_dict, as saved by the upstream trainer.')
    state = {k: v for k, v in state.items() if not k.startswith('smplx_model.')}
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith('smplx_model.')]
    if missing or result.unexpected_keys:
        raise ValueError(f'Checkpoint does not match the released model: missing={missing}, '
                         f'unexpected={result.unexpected_keys}')
    return model


def load_motion_model(config):
    device = torch.device(config.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Motion inference requires CUDA and the PointNet2 extension. See docs/code/setup.md.')
    if not Path(config.load_model_dir).is_file():
        raise FileNotFoundError(f'Motion checkpoint not found: {config.load_model_dir}')
    for path in [config.vposer_path, config.smplx_path]:
        if not Path(path).is_dir():
            raise FileNotFoundError(f'External body-model directory not found: {path}')

    import smplx
    from human_body_prior.tools.model_loader import load_vposer
    from models.sgip import SGIP

    vposer, _ = load_vposer(config.vposer_path, vp_model='snapshot')
    vposer = vposer.to(device)
    body_model = smplx.create(
        config.smplx_path, model_type='smplx', gender='neutral', ext='npz',
        num_pca_comps=12, create_global_orient=True, create_body_pose=True,
        create_betas=True, create_left_hand_pose=True, create_right_hand_pose=True,
        create_expression=True, create_jaw_pose=True, create_leye_pose=True,
        create_reye_pose=True, create_transl=True, num_betas=10,
        num_expression_coeffs=10,
        batch_size=config.batch_size * (config.input_seq_len + config.output_seq_len),
    ).to(device)
    model = SGIP(config, vposer, body_model).to(device)
    load_motion_weights(model, config.load_model_dir, device)
    return model.eval(), device
