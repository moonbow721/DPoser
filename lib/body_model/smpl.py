import torch
import numpy as np
from smplx import SMPL as _SMPL
from smplx import SMPLX as _SMPLX

from lib.body_model import constants
from lib.utils.transforms import rot6d_to_axis_angle

try:
    from smplx.body_models import ModelOutput as SMPLOutput
except Exception as e:
    from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

# from CLIFF (https://github.com/haofanwang/CLIFF), for fitting 2d keypoints only
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(constants.SMPL_REGRESSOR_PATH)
        smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))  # [9, 6890]
        rot6d_poses = torch.tensor(smpl_mean_params['pose'], dtype=torch.float32)  # [144]
        axis_poses = rot6d_to_axis_angle(rot6d_poses.reshape(-1, 6)).reshape(-1)
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


class SMPLX(torch.nn.Module):
    def __init__(self, model_path: str, **kwargs):
        super(SMPLX, self).__init__()
        self.bm = _SMPLX(model_path, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
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

