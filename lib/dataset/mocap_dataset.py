# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import pickle

import cv2
import numpy as np

from torch.utils.data import Dataset
from lib.body_model.body_model import BodyModel
from lib.utils.preprocess import process_image, load_ply
from lib.utils.transforms import estimate_focal_length, rigid_align


class MocapDataset(Dataset):
    def __init__(self, img_bgr_list, detection_list, batchsize=1, device='cuda:0', body_model_path=None):
        self.img_bgr_list = img_bgr_list
        self.detection_list = detection_list
        self.device = device

        # To evaluate EHF
        self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
        if body_model_path is not None:
            self.smplx = BodyModel(bm_path=body_model_path,
                                   num_betas=10,
                                   batch_size=batchsize,
                                   model_type='smplx').to(device)


    def __len__(self):
        return len(self.detection_list)

    def __getitem__(self, idx):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y]
        :param idx:
        :return:
        """
        item = {}
        img_idx = int(self.detection_list[idx][0].item())
        img_bgr = self.img_bgr_list[img_idx]
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        bbox = self.detection_list[idx][1:5]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length
        return item

    def eval_EHF(self, pred_results, gt_ply_path):
        eval_result = {'pa_mpjpe_body': [], 'mpjpe_body': []}
        batchsize = pred_results[0].shape[0]
        if batchsize > 1:
            assert isinstance(gt_ply_path, list) and len(gt_ply_path) == batchsize
            gt_ply_path_list = gt_ply_path
        else:
            gt_ply_path_list = [gt_ply_path]
        pose, betas, camera_translation, reprojection_loss = pred_results
        mesh_out = self.smplx(betas=betas,
                              body_pose=pose[:, 3:66],
                              global_orient=pose[:, :3],
                              trans=camera_translation).v.detach().cpu().numpy()
        for idx, gt_ply in enumerate(gt_ply_path_list):
            mesh_gt = load_ply(gt_ply)
            mesh_gt = np.dot(self.cam_param['R'], mesh_gt.transpose(1, 0)).transpose(1, 0)
            # MPJPE from body joints
            joint_gt_body = np.dot(self.smplx.J_regressor, mesh_gt)[:22]
            joint_out_body = np.dot(self.smplx.J_regressor, mesh_out[idx])[:22]
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            eval_result['pa_mpjpe_body'].append(np.sqrt(np.sum((joint_out_body_align-joint_gt_body) ** 2, axis=1)).mean() * 1000)
            joint_out_body_align = joint_out_body - joint_out_body[self.smplx.J_regressor_idx['pelvis'], None, :] + \
                                   joint_gt_body[self.smplx.J_regressor_idx['pelvis'], None, :]
            eval_result['mpjpe_body'].append(np.sqrt(np.sum((joint_out_body_align-joint_gt_body) ** 2, axis=1)).mean() * 1000)

        return eval_result


    def eval_EHF_multi_hypo(self, pred_results, gt_ply_path):
        #  pred_results: list of results tuple (pose, betas, camera_translation, reprojection_loss)
        eval_result = {'pa_mpjpe_body': float('inf'), 'mpjpe_body': float('inf'), 'pck_body': 0.0}
        threshold = 0.15

        mesh_gt = load_ply(gt_ply_path)
        mesh_gt = np.dot(self.cam_param['R'], mesh_gt.transpose(1, 0)).transpose(1, 0)
        joint_gt_body = np.dot(self.smplx.J_regressor, mesh_gt)[:22]
        pelvis_idx = self.smpl.J_regressor_idx['pelvis']
        gt_pelvis = joint_gt_body[pelvis_idx, None, :]

        for i in range(len(pred_results)):
            pose, betas, camera_translation, reprojection_loss = pred_results[i]

            mesh_out = self.smplx(betas=betas,
                                  body_pose=pose[:, 3:66],
                                  global_orient=pose[:, :3],
                                  trans=camera_translation).v.detach().cpu().numpy()[0]

            joint_out_body = np.dot(self.smplx.J_regressor, mesh_out)[:22]
            joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
            pa_mpjpe = np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1)).mean() * 1000
            eval_result['pa_mpjpe_body'] = min(eval_result['pa_mpjpe_body'], pa_mpjpe)
            joint_out_body_align = joint_out_body - joint_out_body[pelvis_idx, None, :] + gt_pelvis
            mpjpe = np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1)).mean() * 1000
            eval_result['mpjpe_body'] = min(eval_result['mpjpe_body'], mpjpe)
            # Calculate PCK
            distances = np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, axis=1))
            pck = np.mean(distances <= threshold)
            eval_result['pck_body'] = max(eval_result['pck_body'], pck)

        return eval_result


    def print_eval_result(self, eval_result):
        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))
        if 'pck_body' in eval_result:
            print('PCK: %.5f mm' % np.mean(eval_result['pck_body']))
