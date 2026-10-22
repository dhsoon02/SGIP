"""Arguments for the released 6-frame / 10-frame motion model."""

from argparse import ArgumentParser
import os


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise ValueError('must be a positive integer')
    return value


class SGIPConfig(ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_argument('--checkpoint', '--load_model_dir', dest='load_model_dir', required=True,
                          help='Compatible motion-model state_dict (.pth); weights are not bundled.')
        self.add_argument('--device', default='cuda', help='CUDA device, e.g. cuda or cuda:0.')
        self.add_argument('--batch_size', type=positive_int, default=1)
        self.add_argument('--num_workers', type=int, default=0)
        self.add_argument('--seed', type=int, default=7)
        self.add_argument('--deterministic', action='store_true')
        self.add_argument('--dataroot', default=os.environ.get('DATAROOT', 'data/GIMO'))
        self.add_argument('--object_points_name', default='student_object_points.pth',
                          help='Student-selected point cache, relative to SLICES_888s/{split}.')
        self.add_argument('--object_mask_name', default=None)
        self.add_argument('--smplx_path', default=os.environ.get('SMPLX_PATH', 'data/body_models'))
        self.add_argument('--vposer_path', default=os.environ.get('VPOSER_PATH', 'vposer_v1_0'))
        self.add_argument('--sample_points', type=positive_int, default=32768)
        # The upstream GCN and observed-frame splice are fixed to this protocol.
        self.add_argument('--input_seq_len', type=int, choices=[6], default=6)
        self.add_argument('--output_seq_len', type=int, choices=[10], default=10)
        self.add_argument('--fps', type=int, choices=[2], default=2)
        self.add_argument('--dropout', type=float, default=0.0)
        self.add_argument('--motion_n_heads', type=positive_int, default=8)
        self.add_argument('--motion_hidden_dim', type=positive_int, default=256)
        self.add_argument('--motion_n_layers', type=positive_int, default=3)
        self.add_argument('--motion_latent_dim', type=positive_int, default=256)
        self.add_argument('--scene_feats_dim', type=positive_int, default=256)
