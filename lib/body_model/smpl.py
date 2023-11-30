import torch
import numpy as np
from smplx import SMPLX as _SMPLX

from lib.body_model import constants
from lib.utils.transforms import rot6d_to_axis_angle

try:
    from smplx.body_models import ModelOutput as SMPLOutput
except Exception as e:
    from smplx.body_models import SMPLOutput

# from CLIFF (https://github.com/haofanwang/CLIFF), for fitting 2d keypoints only

# Dict containing the joints in numerical order
JOINT_IDS = {constants.JOINT_NAMES[i]: i for i in range(len(constants.JOINT_NAMES))}


'''
class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(constants.SMPL_REGRESSOR_PATH)
        smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))  # [9, 6890]
        self.register_buffer('mean_poses', axis_poses)  # [72]
        self.register_buffer('mean_shape', torch.tensor(smpl_mean_params['shape'], dtype=torch.float32))  # [10]
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output
'''


class SMPLX(torch.nn.Module):
    def __init__(self, model_path: str, **kwargs):
        super(SMPLX, self).__init__()
        self.bm = _SMPLX(model_path, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        # a hack from SMPL to SMPLX openpose joints
        joints[:25] = [55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                       8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                       63, 64, 65]
        # borrow from smpl, as simple initialization
        smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
        rot6d_poses = torch.tensor(smpl_mean_params['pose'], dtype=torch.float32)
        axis_poses = rot6d_to_axis_angle(rot6d_poses.reshape(-1, 6)).reshape(-1)
        self.register_buffer('mean_poses', axis_poses)  # [72]
        self.register_buffer('mean_shape', torch.tensor(smpl_mean_params['shape'], dtype=torch.float32))  # [10]
        self.faces = self.bm.faces_tensor.numpy()
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = self.bm(*args, **kwargs)
        joints = smpl_output.joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output

