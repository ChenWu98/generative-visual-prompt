# Created by Chen Henry Wu
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
from pytorch3d.transforms import (
    so3_relative_angle,
    axis_angle_to_matrix,
    euler_angles_to_matrix,
)

from ..model_utils import requires_grad
from ..lib.decalib.deca import DECA
from ..lib.decalib.utils.config import cfg as deca_cfg


class PoseEnergy(nn.Module):
    def __init__(self, canonical_pose_euler):
        super(PoseEnergy, self).__init__()

        deca_cfg.model.use_tex = False
        deca_cfg.rasterizer_type = "pytorch3d"

        self.preprocess = transforms.Compose(  # Already un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [Resize(224, interpolation=InterpolationMode.BICUBIC)]
        )

        # x: left-right, y: up-down (positive: turn right in our eyes), z: in-out
        self.convention = 'XYZ'
        self.register_buffer(
            "canonical_pose",
            euler_angles_to_matrix(
                euler_angles=torch.FloatTensor(canonical_pose_euler).unsqueeze(0),
                convention=self.convention,
            )
        )
        assert self.canonical_pose.dim() == 3

        self.deca = DECA(config=deca_cfg)

        # Freeze.
        requires_grad(self.deca, False)

    @ staticmethod
    def prepare_inputs(img, z, y=None, alpha=None):
        return {
            'img': img,
            'rel_pose_euler': alpha,
        }

    def forward(self, img, rel_pose_euler=None):
        # Eval mode for the DECA model.
        self.deca.eval()

        # Get target pose.
        rel_pose = euler_angles_to_matrix(
            euler_angles=rel_pose_euler,
            convention=self.convention,
        )
        target_pose = torch.einsum("bxy,byz->bxz", rel_pose, self.canonical_pose)

        # Get pose detected by DECA.
        img = self.preprocess(img)
        code_dict = self.deca.encode(img, use_detail=False)  # When use_detail=True, no_grad is applied in deca.
        global_pose_angle_axis = code_dict['pose'][:, :3]
        global_pose = axis_angle_to_matrix(global_pose_angle_axis)

        try:
            pose_loss = torch.square(
                so3_relative_angle(
                    target_pose,
                    global_pose,
                    eps=1e-3,
                )
            )
        except Exception:
            print(target_pose)
            print(global_pose)
            print(torch.bmm(target_pose, global_pose.permute(0, 2, 1)))
            pose_loss = 0

        return pose_loss

