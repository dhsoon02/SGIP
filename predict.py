"""Predict ten future frames from observed motion and selected scene points."""

from pathlib import Path
from configs.motion import SGIPConfig


def validate_inputs(bundle, config):
    import torch

    required = ('poses_input', 'joints_input', 'scene_points', 'objects', 'object_mask')
    if not isinstance(bundle, dict) or any(key not in bundle for key in required):
        raise ValueError(f'Input must be a tensor dictionary with keys: {required}')
    data = {key: bundle[key] for key in required}
    if any(not isinstance(value, torch.Tensor) for value in data.values()):
        raise ValueError('Every input field must be a tensor.')
    poses = data['poses_input']
    if poses.ndim != 3 or poses.shape[0] == 0 or poses.shape[1:] != (config.input_seq_len, 38):
        raise ValueError('poses_input must have shape [N, 6, 38] with N > 0.')
    n = poses.shape[0]
    joints = data['joints_input']
    if joints.ndim != 4 or joints.shape[:2] != (n, config.input_seq_len) or joints.shape[2] < 23 or joints.shape[3] != 3:
        raise ValueError('joints_input must have shape [N, 6, J, 3], J >= 23.')
    if data['scene_points'].shape != (n, config.sample_points, 3):
        raise ValueError(f'scene_points must have shape [N, {config.sample_points}, 3].')
    objects = data['objects']
    if objects.ndim != 3 or objects.shape[0] != n or objects.shape[1] == 0 or objects.shape[2] != 3:
        raise ValueError('objects must have shape [N, P, 3], P > 0.')
    mask = data['object_mask']
    if mask.shape != (n,) or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('object_mask must have shape [N] with boolean or 0/1 values.')
    for key in required[:-1]:
        if not data[key].is_floating_point() or not torch.isfinite(data[key]).all():
            raise ValueError(f'{key} must contain finite floating-point values.')
        data[key] = data[key].float()
    data['joints_input'] = data['joints_input'][:, :, :23].contiguous()
    data['object_mask'] = mask.bool()
    data['objects'] = data['objects'].clone()
    data['objects'][~data['object_mask']] = 0
    return data


def predict_batches(model, data, config, device):
    import torch

    poses, joints = [], []
    with torch.inference_mode():
        for start in range(0, data['poses_input'].shape[0], config.batch_size):
            batch = {key: value[start:start + config.batch_size].to(device) for key, value in data.items()}
            latent, prediction = model(batch['poses_input'], batch['joints_input'],
                                       batch['scene_points'].contiguous(), batch['objects'], batch['object_mask'])
            poses.append(latent[:, config.input_seq_len:].cpu())
            joints.append(prediction[:, config.input_seq_len:].cpu())
    return {'poses_pred': torch.cat(poses), 'joints_pred': torch.cat(joints)}


def main():
    parser = SGIPConfig(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Observed-input tensor dictionary (.pt).')
    parser.add_argument('--output', required=True, type=Path, help='Output tensor dictionary (.pt).')
    config = parser.parse_args()
    if config.input.resolve() == config.output.resolve():
        parser.error('--output must differ from --input')

    import torch
    from utils.runtime import load_motion_model, seed_everything

    seed_everything(config.seed, config.deterministic)
    data = validate_inputs(torch.load(config.input, map_location='cpu', weights_only=True), config)
    model, device = load_motion_model(config)
    output = predict_batches(model, data, config, device)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, config.output)
    print(f'Saved {output["joints_pred"].shape[0]} predictions to {config.output}')


if __name__ == '__main__':
    main()
