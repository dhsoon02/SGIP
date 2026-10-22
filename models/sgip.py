"""Object-conditioned motion model from the upstream public main path.

The original SIF3D class names and parameter names are retained for checkpoint
compatibility. SGIP is the public alias for this implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCN
from models.base_cross_model import CrossAttentionLayer, PerceiveEncoder, SelfAttentionLayer
from utils.vis_utils import latent_to_joints


def human_centered_scene(xyz, ori, trans):
    bs, seq_len, _ = ori.shape
    sin_ori = torch.sin(ori[:, :, 1]).unsqueeze(-1)
    cos_ori = torch.cos(ori[:, :, 1]).unsqueeze(-1)
    xyz = xyz[:, None, :, :].repeat(1, seq_len, 1, 1) - trans[:, :, None, :]

    matrix = torch.cat(
        [
            torch.cat([sin_ori, torch.zeros_like(sin_ori), -cos_ori], dim=-1),
            torch.cat([torch.zeros_like(sin_ori), torch.ones_like(sin_ori), torch.zeros_like(sin_ori)], dim=-1),
            torch.cat([cos_ori, torch.zeros_like(sin_ori), sin_ori], dim=-1),
        ],
        dim=-1,
    )
    matrix = matrix.view(bs, seq_len, 3, 3)
    return torch.matmul(xyz, matrix)


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim, expansion=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim * expansion)
        self.fc2 = nn.Linear(out_dim * expansion, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out = self.drop(F.gelu(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out


class ObjectBiasScorer(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.query_proj = nn.Linear(feat_dim, feat_dim)
        self.key_proj = nn.Linear(feat_dim, feat_dim)
        self.scale = feat_dim ** -0.5

    def forward(self, object_embedding, scene_feats):
        query = self.query_proj(object_embedding) * self.scale
        key = self.key_proj(scene_feats)
        return torch.matmul(query, key.transpose(1, 2))


class IntentionAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats):
        bs, seq_len, _ = motion.shape
        n_points = scene_feats.shape[1]

        q = self.motion_q(motion) * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        q = q.view(bs, self.num_head, seq_len, -1)
        k = k.view(bs, n_points, self.num_head, -1).transpose(1, 2)
        v = v.view(bs, n_points, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v).view(bs, seq_len, -1)
        return self.out_emb(out)


class SemanticAwareAttention(nn.Module):
    def __init__(self, feat_dim, num_head=6):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_head = num_head
        self.motion_q = nn.Linear(feat_dim, feat_dim)
        self.scene_kv = nn.Linear(feat_dim, feat_dim * 2)
        self.out_emb = nn.Linear(feat_dim, feat_dim)

    def forward(self, motion, scene_feats, spatial_prior):
        bs, seq_len, _ = motion.shape
        n_points = scene_feats.shape[1]

        q = self.motion_q(motion) * (self.feat_dim ** -0.5)
        k, v = torch.split(self.scene_kv(scene_feats), [self.feat_dim, self.feat_dim], dim=-1)
        q = q.view(bs, self.num_head, seq_len, -1)
        k = k.view(bs, n_points, self.num_head, -1).transpose(1, 2)
        v = v.view(bs, n_points, self.num_head, -1).transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))
        att = att + spatial_prior[:, None, :, :]
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_emb(out)


class SCABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.sa_attention = SemanticAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 2, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats, spatial_prior):
        motion_sca = self.motion_encoder(motion)
        sm_feature = self.sa_attention(motion_sca, scene_feats, spatial_prior)
        out = torch.cat([motion_sca, sm_feature], dim=-1)
        return self.mlp(out)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, xyz):
        out = F.relu(self.fc1(xyz))
        return self.fc2(out).squeeze(-1)


class ObjectPointLatentEncoder(nn.Module):
    def __init__(
        self,
        n_latent,
        latent_dim,
        n_cross_att_heads=4,
        n_self_att_heads=8,
        n_self_att_layers=3,
        dropout=0.0,
    ):
        super().__init__()
        self.point_proj = nn.Sequential(
            nn.Linear(3, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.fuse_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.summary_proj = nn.Sequential(
            nn.Linear(6, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.cross_att = CrossAttentionLayer(
            num_q_channels=latent_dim,
            num_kv_channels=latent_dim,
            num_heads=n_cross_att_heads,
            dropout=dropout,
        )
        self.self_att = nn.Sequential(
            *[
                SelfAttentionLayer(
                    num_q_channels=latent_dim,
                    num_heads=n_self_att_heads,
                    dropout=dropout,
                )
                for _ in range(n_self_att_layers)
            ]
        )
        self.latent = nn.Parameter(torch.empty(n_latent, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, object_points):
        point_feats = self.point_proj(object_points)
        global_feats = point_feats.max(dim=1, keepdim=True)[0].expand(-1, point_feats.shape[1], -1)
        point_feats = self.fuse_proj(torch.cat([point_feats, global_feats], dim=-1))

        center = object_points.mean(dim=1)
        extent = object_points.amax(dim=1) - object_points.amin(dim=1)
        summary = self.summary_proj(torch.cat([center, extent], dim=-1)).unsqueeze(1)

        kv_feats = self.dropout(torch.cat([summary, point_feats], dim=1))
        latent = self.latent.unsqueeze(0).expand(object_points.shape[0], -1, -1)
        latent = self.cross_att(latent, kv_feats)
        return self.self_att(latent)


class TIABlock(nn.Module):
    def __init__(self, feat_dim, num_head=4):
        super().__init__()
        self.motion_encoder = SelfAttentionLayer(num_heads=8, num_q_channels=feat_dim, dropout=0.0)
        self.ia_attention = IntentionAwareAttention(feat_dim, num_head)
        self.mlp = Mlp(feat_dim * 3, feat_dim, drop=0.0)

    def forward(self, motion, scene_feats, object_embedding):
        motion_tia = self.motion_encoder(motion)
        sm_feature = self.ia_attention(motion_tia, scene_feats)
        out = torch.cat([motion_tia, sm_feature, object_embedding], dim=-1)
        return self.mlp(out)


class SIF3DObjectPointNoGazeBase(nn.Module):
    def __init__(
        self,
        config,
        vposer,
        smplx_model,
        tia_block_cls,
        use_object_prior_in_sca,
        sca_block_cls=SCABlock,
        use_gt_point_prior_in_sca=False,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.disable_object_conditioning = bool(getattr(config, "student_v0_no_object", False))

        self.vposer = vposer
        self.smplx_model = smplx_model

        from models.pointnet_plus2 import PointNet2SemSegSSGShape

        self.pointnet = PointNet2SemSegSSGShape({"feat_dim": config.scene_feats_dim})
        self.motion_emb = nn.Linear(69 + 38, config.motion_hidden_dim)
        self.motion_encoder = PerceiveEncoder(
            n_input_channels=config.motion_hidden_dim,
            n_latent=config.output_seq_len + config.input_seq_len,
            n_latent_channels=config.motion_latent_dim,
            n_self_att_heads=config.motion_n_heads,
            n_self_att_layers=config.motion_n_layers,
            dropout=config.dropout,
        )

        self.object_encoder = ObjectPointLatentEncoder(
            n_latent=config.output_seq_len + config.input_seq_len,
            latent_dim=config.motion_latent_dim,
            n_cross_att_heads=4,
            n_self_att_heads=config.motion_n_heads,
            n_self_att_layers=config.motion_n_layers,
            dropout=config.dropout,
        )

        self.spatial_att = SpatialPrior()
        self.dist_factor = nn.Parameter(torch.zeros(1))
        self.dir_factor = nn.Parameter(torch.zeros(1))
        self.obj_dir_factor = nn.Parameter(torch.zeros(1))
        self.use_object_prior_in_sca = use_object_prior_in_sca
        self.use_gt_point_prior_in_sca = use_gt_point_prior_in_sca
        self.object_prior_scorer = ObjectBiasScorer(config.motion_latent_dim)
        self.object_prior_factor = nn.Parameter(torch.zeros(1))
        self.gt_prior_factor = nn.Parameter(torch.ones(1))

        self.tia_blocks = nn.ModuleList([tia_block_cls(config.motion_latent_dim, num_head=4) for _ in range(2)])
        self.trajectory_fc = nn.Linear(config.motion_latent_dim, 6)

        self.sca_blocks = nn.ModuleList([sca_block_cls(config.motion_latent_dim, num_head=4) for _ in range(2)])
        self.pose_fc = nn.Linear(config.motion_latent_dim, 32)

        self.joints_fc = nn.Linear(config.motion_latent_dim * 2, 69)
        self.motion_decoder = GCN(config, node_n=69)

        if self.disable_object_conditioning:
            for module in (self.object_encoder, self.object_prior_scorer):
                for param in module.parameters():
                    param.requires_grad_(False)
            self.obj_dir_factor.requires_grad_(False)
            self.object_prior_factor.requires_grad_(False)

    def _horizontal_unit(self, vec):
        return vec / torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)

    def _build_object_dir_prior(self, scene_xyz, object_points, trans):
        object_center = object_points.mean(dim=1)

        human_to_obj = object_center[:, None, [0, 2]] - trans[:, :, [0, 2]]
        human_to_scene = scene_xyz[:, None, :, [0, 2]] - trans[:, :, None, [0, 2]]
        obj_dir = self._horizontal_unit(human_to_obj)
        scene_dir = self._horizontal_unit(human_to_scene)
        return (scene_dir * obj_dir[:, :, None, :]).sum(dim=-1)

    def forward(
        self,
        motions,
        joints,
        scene_xyz,
        objects,
        object_mask,
        gt_object_point_prior=None,
        gt_object_point_prior_mask=None,
        return_features=False,
    ):
        bs, input_seq_len = motions.shape[:2]

        if objects.ndim != 3 or objects.shape[-1] != 3:
            raise ValueError(f"Expected object point cloud with shape [B, P, 3], got {objects.shape}")

        object_points = objects
        object_mask = object_mask.float().view(bs, 1, 1)

        scene_feats, _ = self.pointnet(scene_xyz.repeat(1, 1, 2))
        scene_feats = scene_feats.transpose(1, 2)

        motions = torch.cat([motions, joints.view(bs, input_seq_len, -1)], dim=-1)
        motions = self.motion_emb(motions)
        motions = self.motion_encoder(motions)
        motion_seq_len = motions.shape[1]

        if self.disable_object_conditioning:
            object_embedding = torch.zeros(
                bs,
                motion_seq_len,
                self.config.motion_latent_dim,
                dtype=motions.dtype,
                device=motions.device,
            )
        else:
            object_embedding = self.object_encoder(object_points)
            # Sample-wise mask: zero-out object_embedding for samples whose
            # object selector failed (object_mask=False). Without this, the
            # encoder's bias terms produce a non-trivial embedding from an
            # all-zero point cloud, leaking phantom-object information into
            # the TIA blocks and the trajectory_fc head.
            #
            # object_mask has shape [bs, 1, 1] (after the .view above) and
            # broadcasts cleanly to object_embedding's [bs, T, D] shape.
            object_embedding = object_embedding * object_mask

        out = motions.clone()
        for blk in self.tia_blocks:
            out = blk(out, scene_feats, object_embedding)
        out_tia = out.clone()
        ori, trans = torch.split(self.trajectory_fc(out_tia), [3, 3], dim=-1)

        scene_xyz_normalized = human_centered_scene(scene_xyz, ori, trans)
        dist_from_human = torch.norm(scene_xyz_normalized, dim=-1)
        dist_salience = -((dist_from_human * 1.5) ** 3)
        dir_salience = scene_xyz_normalized[:, :, :, 0] / torch.sqrt(
            (scene_xyz_normalized[:, :, :, [0, 2]] ** 2).sum(dim=-1).clamp_min(1e-6)
        )
        spatial_prior = (
            dist_salience * self.dist_factor
            + dir_salience * self.dir_factor
            + self.spatial_att(scene_xyz_normalized)
        )

        if not self.disable_object_conditioning:
            obj_dir_salience = self._build_object_dir_prior(
                scene_xyz,
                object_points,
                trans,
            )
            object_soft_prior = self.obj_dir_factor * obj_dir_salience * object_mask
            spatial_prior = spatial_prior + object_soft_prior

        if self.use_object_prior_in_sca and not self.disable_object_conditioning:
            object_prior = self.object_prior_factor * self.object_prior_scorer(object_embedding, scene_feats)
            spatial_prior = spatial_prior + object_prior * object_mask

        if self.use_gt_point_prior_in_sca:
            if gt_object_point_prior is None or gt_object_point_prior_mask is None:
                raise ValueError("gt_object_point_prior and gt_object_point_prior_mask are required for this model variant")
            gt_object_point_prior = gt_object_point_prior.float()
            gt_object_point_prior_mask = gt_object_point_prior_mask.float().view(bs, 1, 1)
            gt_prior = gt_object_point_prior.unsqueeze(1).expand(-1, motion_seq_len, -1)
            spatial_prior = spatial_prior + self.gt_prior_factor * gt_prior * gt_object_point_prior_mask

        out = motions.clone()
        for blk in self.sca_blocks:
            out = blk(out, scene_feats, spatial_prior)
        out_sca = out.clone()
        pose = self.pose_fc(out_sca)

        latent = torch.cat([ori, trans, pose], dim=-1)
        recons_joints = latent_to_joints(latent, self.vposer, self.smplx_model)[:, :, :23]
        recons_joints = torch.cat([joints, recons_joints[:, 6:]], dim=1).clone()

        pred_joints_full = self.motion_decoder(recons_joints)[:, :, :23]
        if return_features:
            features = {
                "motion_enc": motions,
                "object_emb": object_embedding,
                "out_tia": out_tia,
                "out_sca": out_sca,
                "ori_trans": torch.cat([ori, trans], dim=-1),
                "pose_latent": pose,
            }
            return latent, pred_joints_full, features
        return latent, pred_joints_full


class SIF3D(SIF3DObjectPointNoGazeBase):
    def __init__(self, config, vposer, smplx_model):
        super().__init__(
            config,
            vposer,
            smplx_model,
            TIABlock,
            use_object_prior_in_sca=False,
            sca_block_cls=SCABlock,
            use_gt_point_prior_in_sca=False,
        )
        self.tia_blocks = nn.ModuleList(
            [TIABlock(config.motion_latent_dim, num_head=4) for _ in range(3)]
        )
        self.sca_blocks = nn.ModuleList(
            [SCABlock(config.motion_latent_dim, num_head=4) for _ in range(3)]
        )

        # v4_nogaze uses only the object-direction prior in SCA.


SGIP = SIF3D
