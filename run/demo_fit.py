'''
# borrow and modify from CLIFF (https://github.com/haofanwang/CLIFF)
'''

import argparse
import json
import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.body_model.fitting_losses import *
from lib.body_model.smpl import SMPLX
from lib.body_model.visual import Renderer
from lib.dataset.mocap_dataset import MocapDataset
from lib.utils.preprocess import compute_bbox
from lib.utils.transforms import cam_crop2full
from .smplify import SMPLify

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-folder', type=str, default='../data/AMASS/amass_processed',
                    help='the folder includes necessary normalizing parameters')
parser.add_argument('--version', type=str, default='version1', help='dataset version')

parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/axis-zscore-400k.pth')
parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                    help='path of SMPLX model')
parser.add_argument('--config-path', type=str,
                    default='configs.subvp.amass_scorefc_continuous.get_config',
                    help='config files to build DPoser')
parser.add_argument('--sde-N', type=int, default=500,
                    help='discrete steps for whole reverse diffusion')
parser.add_argument('--time-strategy', type=str, default='3', choices=['1', '2', '3'],
                    help='random, fix, truncated annealing')

parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--openpose', type=str, default=None, help='Path to .json containing openpose detections')
parser.add_argument('--outdir', type=str, default='./output/test_results/hmr',
                    help='output directory of fitting visualization results')
parser.add_argument('--device', type=str, default='cuda:0')


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    smpl = SMPLX(args.bodymodel_path, batch_size=1).to(device)
    N_POSES = 22  # including root orient

    # load image and 2D keypoints from OpenPose
    orig_img_bgr_all = [cv2.imread(args.img)]
    json_data = json.load(open(args.openpose))
    keypoints = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape((25, 3))

    # # [batch_id, min_x, min_y, max_x, max_y]
    # We observed that the true bbox may be not good choice for people
    # in NOT standing gesture (e.g. sitting, bedding). Using a manual bbox may help the initialization.

    bboxes = compute_bbox(json_data)
    # bboxes = np.array([[0, 400, 100, 1000, 1200]])  # For EHF dataset
    batch_size = len(bboxes)
    assert batch_size == 1, 'we only support single person and single image for this demo'

    mocap_db = MocapDataset(orig_img_bgr_all, bboxes, device=args.device, body_model_path=args.bodymodel_path)
    mocap_data_loader = DataLoader(mocap_db, batch_size=batch_size, num_workers=0)

    for batch in mocap_data_loader:
        norm_img = batch["norm_img"].to(device).float()
        center = batch["center"].to(device).float()
        scale = batch["scale"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)  # [-1, 1]

        kpts = np.zeros((1, 49, 3))
        kpts[0, :25, :] = keypoints
        keypoints = torch.from_numpy(kpts).to(device)

        # Convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)

        # the camera z is also important for initialization
        pred_cam_crop = torch.tensor([[1.3, 0, 0]], device=device).repeat(batch_size, 1)
        init_cam_t = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

        smpl_poses = smpl.mean_poses[:N_POSES * 3].unsqueeze(0).repeat(batch_size, 1).to(device)  # N*72
        init_betas = smpl.mean_shape.unsqueeze(0).repeat(batch_size, 1).to(device)  # N*10
        camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

        pred_output = smpl(betas=init_betas,
                           body_pose=smpl_poses[:, 3:],
                           global_orient=smpl_poses[:, :3],
                           pose2rot=True,
                           transl=init_cam_t)

        flag = True
        if flag:
            # re-project to 2D keypoints on image plane
            pred_keypoints3d = pred_output.joints
            rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
            pred_keypoints2d = perspective_projection(pred_keypoints3d,
                                                      rotation,
                                                      init_cam_t,
                                                      focal_length,
                                                      camera_center)  # (N, 49, 2)

            op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
            op_joints_ind = np.array([constants.JOINT_IDS[joint] for joint in op_joints])

            # visualize GT (Openpose) 2D kpts
            orig_img_bgr = orig_img_bgr_all[0].copy()
            keypoints_gt = json.load(open(args.openpose))
            keypoints_gt = np.array(keypoints_gt['people'][0]['pose_keypoints_2d']).reshape((25, 3))
            kpts = np.zeros((1, 49, 3))
            kpts[0, :25, :] = keypoints_gt
            keypoints_gt = kpts

            for index, (px, py, _) in enumerate(keypoints_gt[0][op_joints_ind]):
                cv2.circle(orig_img_bgr, (int(px), int(py)), 1, [255, 128, 0], 2)
            cv2.imwrite(os.path.join(args.outdir, "kpt2d_gt.jpg"), orig_img_bgr)

            # visualize predicted re-project 2D kpts
            orig_img_bgr = orig_img_bgr_all[0].copy()
            for index, (px, py) in enumerate(pred_keypoints2d[0][op_joints_ind]):
                cv2.circle(orig_img_bgr, (int(px), int(py)), 1, [255, 128, 0], 2)
            cv2.imwrite(os.path.join(args.outdir, "kpt2d.jpg"), orig_img_bgr)

            # calculate re-projection loss
            reprojection_error_op = (keypoints_gt[0][op_joints_ind][:, :2] - pred_keypoints2d[0][
                op_joints_ind].detach().cpu().numpy()) ** 2
            print('initial re-projection loss', reprojection_error_op.sum())

            # visualize predicted mesh
            renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                faces=smpl.faces,
                                same_mesh_color=True)

            front_view = renderer.render_front_view(pred_output.vertices.detach().cpu().numpy(),
                                                    bg_img_rgb=orig_img_bgr_all[0][:, :, ::-1].copy())

            cv2.imwrite(os.path.join(args.outdir, "mesh_init.jpg"), front_view[:, :, ::-1])
            renderer.delete()

        # be careful: the estimated focal_length should be used here instead of the default constant
        smplify = SMPLify(body_model=smpl, step_size=1e-2, batch_size=batch_size, num_iters=100,
                          focal_length=focal_length, args=args)

        results = smplify(smpl_poses.detach(),
                          init_betas.detach(),
                          init_cam_t.detach(),
                          camera_center,
                          keypoints)
        new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
        print('after re-projection loss', new_opt_joint_loss.sum().item())

        with torch.no_grad():
            pred_output = smpl(betas=new_opt_betas,
                               body_pose=new_opt_pose[:, 3:],
                               global_orient=new_opt_pose[:, :3],
                               pose2rot=True,
                               transl=new_opt_cam_t)
            pred_vertices = pred_output.vertices

        renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                            faces=smpl.faces,
                            same_mesh_color=True)
        front_view = renderer.render_front_view(pred_vertices.cpu().numpy(),
                                                bg_img_rgb=orig_img_bgr_all[0][:, :, ::-1].copy())
        cv2.imwrite(os.path.join(args.outdir, "mesh_fit.jpg"), front_view[:, :, ::-1])
