# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import cv2
import numpy as np

from torch.utils.data import Dataset
from lib.body_model.body_model import BodyModel
from lib.utils.preprocess import process_image, load_ply
from lib.utils.transforms import estimate_focal_length, rigid_align


class MocapDataset(Dataset):
    def __init__(self, img_bgr_list, detection_list, device, body_model_path):
        self.img_bgr_list = img_bgr_list
        self.detection_list = detection_list
        self.device = device

        # To evaluate EHF
        self.cam_param = {'R': [-2.98747896, 0.01172457, -0.05704687]}
        self.cam_param['R'], _ = cv2.Rodrigues(np.array(self.cam_param['R']))
        self.smplx = BodyModel(bm_path=body_model_path,
                               num_betas=10,
                               batch_size=1,
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
        pose, betas, camera_translation, reprojection_loss = pred_results
        mesh_gt = load_ply(gt_ply_path)
        mesh_gt = np.dot(self.cam_param['R'], mesh_gt.transpose(1, 0)).transpose(1, 0)
        mesh_out = self.smplx(betas=betas,
                              pose_body=pose[:, 3:66],
                              root_orient=pose[:, :3],
                              trans=camera_translation).v.detach().cpu().numpy()[0]

        # MPJPE from body joints
        joint_gt_body = np.dot(self.smplx.J_regressor, mesh_gt)[:22]
        joint_out_body = np.dot(self.smplx.J_regressor, mesh_out)[:22]
        joint_out_body_align = rigid_align(joint_out_body, joint_gt_body)
        eval_result['pa_mpjpe_body'].append(
            np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)
        joint_out_body_align = joint_out_body - np.dot(self.smplx.J_regressor, mesh_out)[self.smplx.J_regressor_idx['pelvis'], None,
                                    :] + np.dot(self.smplx.J_regressor, mesh_gt)[self.smplx.J_regressor_idx['pelvis'],
                                         None, :]
        eval_result['mpjpe_body'].append(
            np.sqrt(np.sum((joint_out_body_align - joint_gt_body) ** 2, 1)).mean() * 1000)

        # self.print_eval_result(eval_result)
        return eval_result

    def print_eval_result(self, eval_result):
        print('PA MPJPE (Body): %.2f mm' % np.mean(eval_result['pa_mpjpe_body']))
        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))
