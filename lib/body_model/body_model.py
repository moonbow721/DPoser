import pickle

import numpy as np
import torch
import torch.nn as nn
from smplx import SMPL, SMPLH, SMPLX
from smplx import joint_names
from smplx.utils import Struct


class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    from https://github.com/davrempe/humor/blob/main/humor/body_model/body_model.py
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=100,
                 model_type='smplh',
                 regressor_path=None):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        '''

        kwargs = {
            'model_type': model_type,
            'num_betas': num_betas,
            'batch_size': batch_size,
            'num_expression_coeffs': num_expressions,
            'use_pca': False,
            'flat_hand_mean': False,
        }

        assert (model_type in ['smpl', 'smplh', 'smplx'])
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate(
                    [data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                    axis=-1)  # super hacky way to let smplh use 16-size beta
            kwargs['data_struct'] = data_struct
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

        if regressor_path is not None:
            for model_path in regressor_path:
                if 'SMPLX_to_J14.pkl' in model_path:
                    with open(model_path, 'rb') as f:
                        # Hand4Whole use it to evalute the EHF dataset
                        self.j14_regressor = pickle.load(f, encoding='latin1')
                elif 'J_regressor_h36m.npy' in model_path:
                    # Use it to evaluate GFPose trained on H36M
                    self.j17_regressor = np.load(model_path)

        self.J_regressor = self.bm.J_regressor.numpy()
        self.J_regressor_idx = {'pelvis': 0, 'lwrist': 20, 'rwrist': 21, 'neck': 12}

    def forward(self, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None,
                jaw_pose=None, eye_poses=None, expression=None, betas=None, trans=None, dmpls=None,
                whole_body_params=None, return_dict=False, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        assert (dmpls is None)
        assert 'pose_body' not in kwargs, 'use body_pose instead of pose_body'

        if whole_body_params is not None:  # [batchsize, 63+90+3+100], body, two_hands, jaw, expression
            assert (self.model_type == 'smplx')
            body_pose = whole_body_params[:, :63]
            left_hand_pose = whole_body_params[:, 63:63 + 45]
            right_hand_pose = whole_body_params[:, 63 + 45:63 + 45 + 45]
            jaw_pose = whole_body_params[:, 63 + 90:63 + 90 + 3]
            expression = whole_body_params[:, 63 + 90 + 3:]

        # parameters of SMPL should not be updated
        out_obj = self.bm(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=trans,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=None if eye_poses is None else eye_poses[:, :3],
            reye_pose=None if eye_poses is None else eye_poses[:, 3:],
            return_full_pose=True,
            **kwargs
        )

        out = {
            'v': out_obj.vertices,
            'f': self.bm.faces_tensor,
            'betas': out_obj.betas,
            'Jtr': out_obj.joints,
            'body_joints': out_obj.joints[:22],  # only body joints
            'body_pose': out_obj.body_pose,
            'full_pose': out_obj.full_pose
        }
        if self.model_type in ['smplh', 'smplx']:
            out['hand_poses'] = torch.cat([out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1)
        if self.model_type == 'smplx':
            out['jaw_pose'] = out_obj.jaw_pose
            out['eye_poses'] = eye_poses

        if not return_dict:
            out = Struct(**out)

        return out
