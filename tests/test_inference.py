from types import SimpleNamespace

import pytest
import torch

from configs.motion import SGIPConfig
from predict import predict_batches, validate_inputs
from utils.metrics import motion_errors
from utils.runtime import load_motion_weights


def config():
    return SimpleNamespace(input_seq_len=6, output_seq_len=10, sample_points=32, batch_size=2)


def inputs(n=3):
    return {
        'poses_input': torch.randn(n, 6, 38),
        'joints_input': torch.randn(n, 6, 127, 3),
        'scene_points': torch.randn(n, 32, 3),
        'objects': torch.randn(n, 8, 3),
        'object_mask': torch.tensor(([True, False, True] * n)[:n]),
    }


def test_inference_requires_no_future_labels_and_masks_invalid_objects():
    original = inputs()
    data = validate_inputs(original, config())
    assert data['joints_input'].shape == (3, 6, 23, 3)
    assert data['objects'][1].count_nonzero() == 0
    assert original['objects'][1].count_nonzero() > 0

    class Model(torch.nn.Module):
        def forward(self, motion, joints, scene, objects, mask):
            # A deterministic stand-in exercises batching and the public output contract.
            assert joints.shape[1:] == (6, 23, 3)
            n = motion.shape[0]
            latent = torch.arange(16).view(1, 16, 1).expand(n, 16, 38).float()
            full_joints = torch.arange(16).view(1, 16, 1, 1).expand(n, 16, 23, 3).float()
            return latent, full_joints

    prediction = predict_batches(Model(), data, config(), torch.device('cpu'))
    assert prediction['poses_pred'].shape == (3, 10, 38)
    assert prediction['joints_pred'].shape == (3, 10, 23, 3)
    assert prediction['joints_pred'][:, 0, 0, 0].tolist() == [6, 6, 6]
    assert prediction['joints_pred'][:, -1, 0, 0].tolist() == [15, 15, 15]


@pytest.mark.parametrize('field', ['poses_input', 'joints_input', 'scene_points', 'objects'])
def test_nonfinite_inputs_fail(field):
    data = inputs()
    data[field].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        validate_inputs(data, config())


def test_bad_shapes_fail_before_cuda_setup():
    data = inputs()
    data['joints_input'] = torch.zeros(3, 6, 22, 3)
    with pytest.raises(ValueError, match='joints_input'):
        validate_inputs(data, config())
    data = inputs()
    data['object_mask'] = torch.tensor([1, 2, 0])
    with pytest.raises(ValueError, match='object_mask'):
        validate_inputs(data, config())


def test_metrics_use_future_frames_and_separate_translation_from_pose():
    observed = torch.zeros(2, 6, 23, 3)
    target = torch.zeros(2, 10, 23, 3)
    prediction = torch.zeros(2, 16, 23, 3)
    prediction[:, :6] = 100  # Observed-frame reconstruction must not affect future metrics.
    prediction[0, 6:, :, 0] = 2  # Uniform world translation: root error, no relative-pose error.
    prediction[1, -1, 1, 0] = 23  # A single relative joint displacement at the destination.
    errors = motion_errors(prediction, observed, target)
    assert torch.allclose(errors['traj_path'], torch.tensor([2., 0.]))
    assert torch.allclose(errors['traj_dest'], torch.tensor([2., 0.]))
    assert torch.allclose(errors['mpjpe_path'], torch.tensor([0., 0.1]))
    assert torch.allclose(errors['mpjpe_dest'], torch.tensor([0., 1.]))


def test_checkpoint_keeps_learned_vposer_but_not_smplx_batch_state(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.motion = torch.nn.Linear(2, 2)
            self.vposer = torch.nn.Linear(2, 2)
            self.smplx_model = torch.nn.Linear(2, 2)

    source, target = Model(), Model()
    initial_body = target.smplx_model.weight.detach().clone()
    path = tmp_path / 'weights.pth'
    torch.save(source.state_dict(), path)
    load_motion_weights(target, path, 'cpu')
    assert torch.equal(source.motion.weight, target.motion.weight)
    assert torch.equal(source.vposer.weight, target.vposer.weight)
    assert torch.equal(initial_body, target.smplx_model.weight)
    state = source.state_dict()
    del state['motion.weight']
    torch.save(state, path)
    with pytest.raises(ValueError, match='missing='):
        load_motion_weights(target, path, 'cpu')


def test_unsupported_temporal_protocol_is_rejected():
    with pytest.raises(SystemExit):
        SGIPConfig().parse_args(['--checkpoint', 'unused.pth', '--input_seq_len', '8'])
