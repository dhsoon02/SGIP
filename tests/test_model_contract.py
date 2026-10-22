"""Exercise the released core without CUDA extensions or licensed body assets.

PointNet2 and both body models are stand-ins here. These checks do not validate
their numerical behavior or a trained SGIP checkpoint.
"""

import sys
from types import ModuleType, SimpleNamespace

import torch

from models.sgip import SGIP


class PointBackbone(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.projection = torch.nn.Linear(6, hparams['feat_dim'])

    def forward(self, points):
        features = self.projection(points)
        return features.transpose(1, 2), features.mean(dim=1)


class PoseDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(32, 63)

    def decode(self, latent, output_type):
        assert output_type == 'aa'
        return self.projection(latent)


class BodyModel(torch.nn.Module):
    def forward(self, body_pose, transl, **kwargs):
        # Preserve the body API's shape and dependence on the latent pose.
        joints = body_pose.reshape(-1, 21, 3).repeat(1, 7, 1)[:, :127]
        return SimpleNamespace(joints=joints + transl[:, None])


def test_core_forward_shapes_object_mask_and_partial_batch(monkeypatch):
    backbone_module = ModuleType('models.pointnet_plus2')
    backbone_module.PointNet2SemSegSSGShape = PointBackbone
    monkeypatch.setitem(sys.modules, 'models.pointnet_plus2', backbone_module)
    config = SimpleNamespace(
        input_seq_len=6, output_seq_len=10, scene_feats_dim=32,
        motion_hidden_dim=32, motion_latent_dim=32, motion_n_heads=8,
        motion_n_layers=1, dropout=0.0,
    )
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = SGIP(config, PoseDecoder(), BodyModel()).eval()
        inputs = (
            torch.randn(2, 6, 38), torch.randn(2, 6, 23, 3),
            torch.randn(2, 32, 3), torch.randn(2, 8, 3),
            torch.tensor([True, False]),
        )
        with torch.inference_mode():
            latent, joints, features = model(*inputs, return_features=True)
            assert latent.shape == (2, 16, 38)
            assert joints.shape == (2, 16, 23, 3)
            assert torch.isfinite(latent).all() and torch.isfinite(joints).all()
            assert features['object_emb'][1].count_nonzero() == 0
            changed = list(inputs)
            changed[3] = inputs[3].clone()
            changed[3][1] += 100
            changed_latent, changed_joints = model(*changed)
            torch.testing.assert_close(changed_latent, latent)
            torch.testing.assert_close(changed_joints, joints)
            # A final one-sample batch must work after a larger batch.
            last_latent, last_joints = model(*(value[1:] for value in inputs))
            torch.testing.assert_close(last_latent, latent[1:])
            torch.testing.assert_close(last_joints, joints[1:])
            assert model.smplx_model.batch_size == 16
    finally:
        torch.set_num_threads(original_threads)
