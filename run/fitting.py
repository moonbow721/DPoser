'''
# borrow and modify from CLIFF (https://github.com/haofanwang/CLIFF)
'''

import argparse
import json
import os.path
from glob import glob

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.body_model.fitting_losses import *
from lib.body_model.smpl import SMPLX
from lib.body_model.visual import Renderer
from lib.dataset.mocap_dataset import MocapDataset
from lib.utils.preprocess import compute_bbox
from lib.utils.transforms import cam_crop2full
from .smplify import SMPLify


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-folder', type=str, default='./data/AMASS/amass_processed',
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

parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
parser.add_argument('--outdir', type=str, default='lifting_results/output',
                    help='output directory of fitting visualization results')
parser.add_argument('--device', type=str, default='cuda:0')


if __name__ == '__main__':
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device

    # Load SMPLX model
    smpl = SMPLX(args.bodymodel_path, batch_size=1).to(device)
    N_POSES = 22  # or 24 for smpl, including root orient

    img_paths = sorted(glob(f"{args.data_dir}/*_img.jpg"))
    json_paths = sorted(glob(f"{args.data_dir}/*_2Djnt.json"))
    gt_ply_paths = sorted(glob(f"{args.data_dir}/*_align.ply"))
    total_length = len(img_paths)
    all_eval_results = {'pa_mpjpe_body': [], 'mpjpe_body': []}
    for img_path, json_path, gt_ply_path in tqdm(zip(img_paths, json_paths, gt_ply_paths), desc='Dataset', total=total_length):
        base_name = os.path.basename(img_path)
        img_name, _ = os.path.splitext(base_name)
        # load image and 2D keypoints from OpenPose
        orig_img_bgr_all = [cv2.imread(img_path)]
        json_data = json.load(open(json_path))
        keypoints = np.array(json_data['people'][0]['pose_keypoints_2d']).reshape((25, 3))

        bboxes = compute_bbox(json_data)

        # [batch_id, min_x, min_y, max_x, max_y]
        bend_init = True if bboxes[0, 2] > 400 else False
        bboxes = np.array([[0, 400, 100, 1000, 1200]])
        batch_size = len(bboxes)
        assert batch_size == 1, 'we only support single person and single image now'

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

            pred_cam_crop = torch.tensor([[0.9, 0, 0]], device=device).repeat(batch_size, 1)
            init_cam_t = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

            smpl_poses = smpl.mean_poses[:N_POSES*3].unsqueeze(0).repeat(batch_size, 1).to(device)  # N*72
            if bend_init:
                bend_pose = torch.from_numpy(np.load(constants.BEND_POSE_PATH)['pose']).to(smpl_poses.device)
                smpl_poses = bend_pose[:, :N_POSES*3]

            init_betas = smpl.mean_shape.unsqueeze(0).repeat(batch_size, 1).to(device)  # N*10
            camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

            pred_output = smpl(betas=init_betas,
                               body_pose=smpl_poses[:, 3:],
                               global_orient=smpl_poses[:, :3],
                               pose2rot=True,
                               transl=init_cam_t)

            # be careful: the estimated focal_length should be used here instead of the default constant
            smplify = SMPLify(body_model=smpl, step_size=1e-2, batch_size=batch_size, num_iters=100, focal_length=focal_length, args=args)

            results = smplify(smpl_poses.detach(),
                              init_betas.detach(),
                              init_cam_t.detach(),
                              camera_center,
                              keypoints)

            new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
            # print('after re-projection loss', new_opt_joint_loss.sum().item())
            batch_results = mocap_db.eval_EHF(results, gt_ply_path)
            mocap_db.print_eval_result(batch_results)
            all_eval_results['pa_mpjpe_body'].extend(batch_results['pa_mpjpe_body'])
            all_eval_results['mpjpe_body'].extend(batch_results['mpjpe_body'])
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
            renderer.delete()
            cv2.imwrite(os.path.join(args.outdir, f"{img_name}_mesh_fit.png"), front_view[:, :, ::-1])

    print('results on whole dataset:')
    mocap_db.print_eval_result(all_eval_results)