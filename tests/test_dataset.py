from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from datasets.gimo import EgoEvalDataset, resolve_split_cache_path


def make_cache(root):
    split = root / 'SLICES_888s' / 'test'
    split.mkdir(parents=True)
    pd.DataFrame([
        dict(sequence_path='seq0', training=0, scene='scene', transformation='transform.json', start_frame=0, end_frame=15),
        dict(sequence_path='seq1', training=0, scene='scene', transformation='transform.json', start_frame=16, end_frame=31),
    ]).to_csv(root / 'dataset.csv', index=False)
    tensors = {
        'student_object_points.pth': torch.ones(2, 8, 3),
        'student_object_points_mask.pth': torch.tensor([True, False]),
        'poses_input.pth': torch.zeros(2, 6, 38),
        'poses_label.pth': torch.zeros(2, 10, 38),
        'joints_input.pth': torch.zeros(2, 6, 23, 3),
        'joints_label.pth': torch.zeros(2, 10, 23, 3),
        'scene_points_32.pth': torch.zeros(2, 32, 3),
    }
    for name, tensor in tensors.items():
        torch.save(tensor, split / name)
    return SimpleNamespace(dataroot=str(root), input_seq_len=6, output_seq_len=10, fps=2,
                           sample_points=32, object_points_name='student_object_points.pth', object_mask_name=None)


def test_cache_loading_needs_no_teacher_or_gt_object_cache(tmp_path):
    dataset = EgoEvalDataset(make_cache(tmp_path))
    assert len(dataset) == 2
    assert len(dataset[0]) == 11
    assert dataset[1][0].count_nonzero() == 0
    assert not dataset[0][3]
    assert dataset[1][-2:] == ('seq1', 'scene')


def test_metadata_length_mismatch_is_not_silently_accepted(tmp_path):
    config = make_cache(tmp_path)
    csv = tmp_path / 'dataset.csv'
    pd.read_csv(csv).iloc[:1].to_csv(csv, index=False)
    with pytest.raises(ValueError, match='dataset.csv'):
        EgoEvalDataset(config)


def test_label_length_mismatch_is_reported(tmp_path):
    config = make_cache(tmp_path)
    torch.save(torch.zeros(1, 10, 23, 3), tmp_path / 'SLICES_888s/test/joints_label.pth')
    with pytest.raises(ValueError, match='joints_label length'):
        EgoEvalDataset(config)


def test_split_placeholder_supports_absolute_and_relative_cache_paths(tmp_path):
    assert resolve_split_cache_path(str(tmp_path), 'objects_{split}.pth', 'test') == str(tmp_path / 'objects_test.pth')
    assert resolve_split_cache_path(str(tmp_path), '/data/{split}/objects.pth', 'test') == '/data/test/objects.pth'
