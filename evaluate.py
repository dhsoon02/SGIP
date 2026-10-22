"""Evaluate a compatible motion checkpoint on prepared GIMO-format caches."""

import json
from pathlib import Path

from configs.motion import SGIPConfig


def main():
    parser = SGIPConfig(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON metrics file.')
    config = parser.parse_args()
    if config.num_workers < 0:
        parser.error('--num_workers must be non-negative')

    import torch
    from torch.utils.data import DataLoader
    from datasets.gimo import EgoEvalDataset
    from utils.metrics import motion_errors
    from utils.runtime import load_motion_model, seed_everything

    seed_everything(config.seed, config.deterministic)
    dataset = EgoEvalDataset(config, train=False)
    if not len(dataset):
        raise ValueError('The evaluation split contains no samples.')
    model, device = load_motion_model(config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers, drop_last=False)
    totals = dict.fromkeys(['traj_path', 'traj_dest', 'mpjpe_path', 'mpjpe_dest'], 0.0)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            objects, mask, _, _, motion, _, joints, target, scene, _, _ = batch
            # Labels are used only by the metric calculation below.
            motion, joints, target, scene, objects, mask = [
                value.to(device) for value in [motion, joints[:, :, :23], target[:, :, :23],
                                              scene, objects, mask]
            ]
            _, prediction = model(motion, joints, scene.contiguous(), objects, mask)
            errors = motion_errors(prediction, joints, target)
            for key, values in errors.items():
                totals[key] += values.sum().item()
            count += motion.shape[0]
    result = {'samples': count, 'units': 'input_coordinate_units',
              **{key: total / count for key, total in totals.items()}}
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if config.output:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(text + '\n')


if __name__ == '__main__':
    main()
