import argparse
import json
import os.path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.body_model import constants
from lib.body_model.fitting_losses import *
from lib.body_model.joint_mapping import mmpose_to_openpose, vitpose_to_openpose
from lib.body_model.smpl import SMPLX
from lib.body_model.visual import Renderer, vis_keypoints_with_skeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.utils.preprocess import compute_bbox
from lib.utils.transforms import cam_crop2full
from .smplify import SMPLify

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='DPoser', choices=['DPoser', 'VPoser', 'GMM', 'Posendf', 'None'],
                    help='Our prior model or competitors')
parser.add_argument('--kpts', type=str, default='vitpose', choices=['mmpose', 'vitpose', 'openpose'])


parser.add_argument('--dataset-folder', type=str,
                    default='../data/human/Bodydataset/amass_processed', help='dataset root')
parser.add_argument('--version', type=str, default='version1', help='dataset version')
parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                    help='path of SMPLX model')

parser.add_argument('--ckpt-path', type=str,
                    default='./pretrained_models/amass/BaseMLP/epoch=36-step=150000-val_mpjpe=38.17.ckpt',
                    help='load trained diffusion model for DPoser')
parser.add_argument('--config-path', type=str,
                    default='configs.body.subvp.timefc.get_config',
                    help='config files to build DPoser')
parser.add_argument('--time-strategy', type=str, default='3', choices=['1', '2', '3'],
                    help='random, fix, truncated annealing')

parser.add_argument('--img', type=str, required=True, help='Path to input image')
parser.add_argument('--kpt_path', type=str, default=None, help='Path to .json containing kpts detections')
parser.add_argument('--init_camera', type=str, default='optimized', choices=['fixed', 'optimized'])

parser.add_argument('--outdir', type=str, default='./output/test_results/hmr',
                    help='output directory of fitting visualization results')
parser.add_argument('--device', type=str, default='cuda:0')


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    smpl = SMPLX(args.bodymodel_path, batch_size=1).to(device)
    N_POSES = 22  # including root orient

    # load image and 2D keypoints
    img_bgr = cv2.imread(args.img)
    img_name = os.path.basename(args.img).split('.')[0]
    json_data = json.load(open(args.kpt_path))
    if args.kpts == 'openpose':
        keypoints = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape((25, 3))
    elif args.kpts == 'mmpose':
        mm_keypoints = np.array(json_data[0]['keypoints'])
        keypoint_scores = np.array(json_data[0]['keypoint_scores'])
        keypoints = mmpose_to_openpose(mm_keypoints, keypoint_scores)[:25]
    elif args.kpts == 'vitpose':
        vit_keypoints = np.array(json_data[0]['keypoints'])
        keypoints = vitpose_to_openpose(vit_keypoints)[:25]
    else:
        raise NotImplementedError

    # # [batch_id, min_x, min_y, max_x, max_y]
    # We observed that the true bbox may be not good choice for people
    # in NOT standing gesture (e.g. sitting, bedding). Using a adjusted bbox may help the initialization.

    bboxes = compute_bbox([keypoints])
    ratio = (bboxes[0, 3] - bboxes[0, 1]) / (bboxes[0, 4] - bboxes[0, 2])
    bend_init = ratio > 0.75
    if bend_init:
        bboxes[0, 2] /= 3
        print('The person is not standing, use bend pose as initialization')
    batch_size = len(bboxes)
    assert batch_size == 1, 'we only support single person and single image for this demo'

    mocap_db = MocapDataset([img_bgr], bboxes, device=args.device, body_model_path=args.bodymodel_path)
    mocap_data_loader = DataLoader(mocap_db, batch_size=batch_size, num_workers=0)

    for batch in mocap_data_loader:
        norm_img = batch["norm_img"].to(device).float()
        img_h = batch["img_h"].to(device).float()
        img_w = batch["img_w"].to(device).float()
        focal_length = batch["focal_length"].to(device).float()

        kpts = np.zeros((1, 49, 3))
        kpts[0, :25, :] = keypoints
        keypoints_tensor = torch.from_numpy(kpts).to(device)

        init_betas = smpl.mean_shape.unsqueeze(0).repeat(batch_size, 1).to(device)  # N*10
        camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

        smpl_poses = smpl.mean_poses[:N_POSES * 3].unsqueeze(0).repeat(batch_size, 1).to(device)  # N*72
        if bend_init:
            bend_pose = torch.from_numpy(np.load(constants.BEND_POSE_PATH)['pose']).to(smpl_poses.device)
            smpl_poses[:, 3:] = bend_pose[:, 3:N_POSES * 3]

        # Convert the camera parameters from the crop camera to the full camera
        if args.init_camera == 'fixed' or bend_init:
            center = batch["center"].to(device).float()
            scale = batch["scale"].to(device).float()
            full_img_shape = torch.stack((img_h, img_w), dim=-1)
            pred_cam_crop = torch.tensor([[0.9, 0, 0]], device=device).repeat(batch_size, 1)
            init_cam_t = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
        else:
            # This method obtains very bad initialization for bend pose, why?
            init_joints_3d = smpl(betas=init_betas,
                                  body_pose=smpl_poses[:, 3:],
                                  global_orient=smpl_poses[:, :3], ).joints
            init_cam_t = guess_init(init_joints_3d[:, :25], keypoints_tensor[:, :25], focal_length, part='body')

        init_vertices = smpl(betas=init_betas,
                             body_pose=smpl_poses[:, 3:],
                             global_orient=smpl_poses[:, :3],
                             transl=init_cam_t).vertices

        # be careful: the estimated focal_length should be used here instead of the default constant
        smplify = SMPLify(body_model=smpl, step_size=1e-2, batch_size=batch_size, num_iters=100,
                          focal_length=focal_length, args=args)

        results = smplify(smpl_poses.detach(),
                          init_betas.detach(),
                          init_cam_t.detach(),
                          camera_center,
                          keypoints_tensor)
        new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results

        print('after re-projection loss', new_opt_joint_loss.sum().item())

        with torch.no_grad():
            pred_output = smpl(betas=new_opt_betas,
                               body_pose=new_opt_pose[:, 3:],
                               global_orient=new_opt_pose[:, :3],
                               pose2rot=True,
                               transl=new_opt_cam_t)
            pred_vertices = pred_output.vertices

        enable_visual = True
        if enable_visual:
            # re-project to 2D keypoints on image plane
            pred_keypoints3d = pred_output.joints
            img_with_kpts = vis_keypoints_with_skeleton(img_bgr, keypoints, kp_thresh=0.1, radius=2)
            cv2.imwrite(os.path.join(args.outdir, f"{img_name}_kpt2d_gt.jpg"), img_with_kpts)
            rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
            projected_joints = perspective_projection(pred_keypoints3d, rotation, new_opt_cam_t,
                                                      focal_length, camera_center).detach().cpu().numpy()
            dummy_confidence = np.ones((projected_joints.shape[0], projected_joints.shape[1], 1))
            projected_joints = np.concatenate([projected_joints, dummy_confidence], axis=-1)
            img_with_kpts = vis_keypoints_with_skeleton(img_bgr, projected_joints[0, :25], kp_thresh=0.1, radius=2)
            cv2.imwrite(os.path.join(args.outdir, f"{img_name}_kpt2d_pred.jpg"), img_with_kpts)

            # visualize predicted mesh
            renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                faces=smpl.faces,
                                same_mesh_color=True)
            front_view = renderer.render_front_view(init_vertices.detach().cpu().numpy(),
                                                    bg_img_rgb=img_bgr[:, :, ::-1].copy())
            renderer.delete()
            cv2.imwrite(os.path.join(args.outdir, f"{img_name}_mesh_init.jpg"), front_view[:, :, ::-1])

            renderer = Renderer(focal_length=focal_length, img_w=img_w, img_h=img_h,
                                faces=smpl.faces,
                                same_mesh_color=True)
            front_view = renderer.render_front_view(pred_vertices.cpu().numpy(),
                                                    bg_img_rgb=img_bgr[:, :, ::-1].copy())
            renderer.delete()
            cv2.imwrite(os.path.join(args.outdir, f"{img_name}_mesh_fit.png"), front_view[:, :, ::-1])
