import torch


def latent_to_joints(pose_latent, vposer, smplx_model):
    bs, T, c = pose_latent.shape
    batch_size = bs * T
    body_pose = vposer.decode(pose_latent[:, :, 6:], output_type='aa').view(-1, 63)
    dtype = pose_latent.dtype
    device = pose_latent.device
    zeros = {
        "betas": torch.zeros(batch_size, getattr(smplx_model, "num_betas", 10), dtype=dtype, device=device),
        "left_hand_pose": torch.zeros(batch_size, getattr(smplx_model, "num_pca_comps", 12), dtype=dtype, device=device),
        "right_hand_pose": torch.zeros(batch_size, getattr(smplx_model, "num_pca_comps", 12), dtype=dtype, device=device),
        "expression": torch.zeros(batch_size, getattr(smplx_model, "num_expression_coeffs", 10), dtype=dtype, device=device),
        "jaw_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
        "leye_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
        "reye_pose": torch.zeros(batch_size, 3, dtype=dtype, device=device),
    }
    # The SMPL-X package repeats landmark barycentric coordinates using
    # smplx_model.batch_size instead of the actual forward batch size.
    # Keep it in sync so the final partial validation batch can be evaluated.
    smplx_model.batch_size = batch_size
    smplx_output = smplx_model(return_verts=True, body_pose=body_pose,
                               global_orient=pose_latent[:, :, :3].view(-1, 3),
                               transl=pose_latent[:, :, 3:6].view(-1, 3),
                               pose_embedding=pose_latent[:, :, 6:].view(-1, 32),
                               **zeros)
    joints = smplx_output.joints
    return joints.reshape(bs, T, 127, 3)
