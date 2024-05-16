import argparse
import json
import os.path
from glob import glob
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

sys.path.insert(0, '/data3/ljz24/projects/3d/DPoser')
from lib.body_model import constants
from lib.body_model.fitting_losses import perspective_projection, guess_init
from lib.body_model.joint_mapping import mmpose_to_openpose, vitpose_to_openpose
from lib.body_model.smpl import SMPLX
from lib.body_model.visual import Renderer, vis_keypoints_with_skeleton
from lib.dataset.mocap_dataset import MocapDataset
from lib.utils.preprocess import compute_bbox
from lib.utils.transforms import cam_crop2full
from .smplify import SMPLify

from lib2.models.gan import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--prior', type=str, default='generator', choices=['generator', 'discriminator', 'None'],)
parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                    help='path of SMPLX model')

parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
parser.add_argument('--kpts', type=str, default='openpose', choices=['mmpose', 'vitpose', 'openpose'])
parser.add_argument('--init_camera', type=str, default='fixed', choices=['fixed', 'optimized'])
parser.add_argument('--outdir', type=str, default='./lifting_results/output',
                    help='output directory of fitting visualization results')
parser.add_argument('--interpenetration', '-i', action='store_true', help='enable interpenetration penalty')
parser.add_argument('--gt-intrinsic', action='store_true', )
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--device', type=str, default='cuda:0')


def invert_to_latent(net, target_pose, latent_size, device, lr=0.01, num_iterations=1000,
                     body_model=None):
    """
    Inverts target pose parameters to the corresponding latent vector.

    :param net: Trained GAN Generator
    :param target_pose: Target pose parameters to invert
    :param latent_size: Size of the latent vector
    :param device: Device to run the computations on
    :param lr: Learning rate for the optimizer
    :param num_iterations: Number of iterations for the optimization
    :return: Latent vector corresponding to the target pose
    """
    target_pose = target_pose.to(device)
    latent_vector = torch.randn(target_pose.size(0), latent_size).to(device)
    latent_vector.requires_grad = True

    optimizer = optim.Adam([latent_vector], lr=lr)

    for iteration in range(num_iterations):
        optimizer.zero_grad()
        generated_pose = net(latent_vector, output_type='aa').reshape(target_pose.size(0), -1)
        if body_model is not None:
            body_out = body_model(body_pose=generated_pose)
            generated_vert, generated_joint = body_out.vertices, body_out.joints
            body_out = body_model(body_pose=target_pose)
            target_vert, target_joint = body_out.vertices, body_out.joints
            loss = 10*torch.nn.functional.mse_loss(generated_vert, target_vert) + \
                     torch.nn.functional.mse_loss(generated_joint, target_joint)
        else:
            loss = torch.nn.functional.mse_loss(generated_pose, target_pose)
        loss.backward()
        optimizer.step()

    return latent_vector.detach()


latent_size = 32
num_joints = 21


if __name__ == '__main__':
    torch.manual_seed(42)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = args.device
    enable_visual = False

    net = Generator(latentD=latent_size, num_joints=num_joints)
    ckpt = './output/ganS/gan_S_amass/ckpt_0300.pth'  # path to pretrained generative model
    ckpt = torch.load(ckpt, map_location='cpu')['generator_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)

    batch_size = args.batch_size
    # Load SMPLX model
    smpl = SMPLX(args.bodymodel_path, batch_size=batch_size).to(device)
    N_POSES = 22

    img_paths = sorted(glob(f"{args.data_dir}/*_img.jpg"))
    gt_ply_paths = sorted(glob(f"{args.data_dir}/*_align.ply"))
    if args.kpts == 'openpose':
        json_paths = sorted(glob(f"{args.data_dir}/*_2Djnt.json"))
    elif args.kpts == 'mmpose':
        json_paths = sorted(glob(f"{args.data_dir}/mmpose_keypoints/predictions/*_img.json"))
    elif args.kpts == 'vitpose':
        json_paths = sorted(glob(f"{args.data_dir}/vitpose_keypoints/predictions/*_img.json"))
    else:
        raise NotImplementedError

    img_names = [Path(path).stem for path in img_paths]
    total_length = len(img_paths)
    current_idx, batch_keypoints, batch_img, batch_ply = 0, [], [], []
    all_eval_results = {'pa_mpjpe_body': [], 'mpjpe_body': []}
    for img_path, json_path, gt_ply_path in tqdm(zip(img_paths, json_paths, gt_ply_paths), desc='Dataset',
                                                 total=total_length):
        base_name = os.path.basename(img_path)
        img_name, _ = os.path.splitext(base_name)
        # load image and 2D keypoints
        img_bgr = cv2.imread(img_path)
        json_data = json.load(open(json_path))
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
        batch_keypoints.append(keypoints)
        batch_img.append(img_bgr)
        batch_ply.append(gt_ply_path)
        if len(batch_keypoints) < batch_size:
            continue
        bboxes = compute_bbox(batch_keypoints)
        keypoints = np.array(batch_keypoints)
        print('batch keypoints:', keypoints.shape)
        # [batch_id, min_x, min_y, max_x, max_y]
        bend_init = torch.tensor([bboxes[batch_id, 2] > 400 for batch_id in range(batch_size)], device=device)
        bboxes = [np.array([batch_id, 400, 100, 1000, 1200]) for batch_id in range(batch_size)]

        assert len(bboxes) == batch_size
        mocap_db = MocapDataset(batch_img, bboxes, batch_size, args.device, body_model_path=args.bodymodel_path)
        mocap_data_loader = DataLoader(mocap_db, batch_size=batch_size, num_workers=0)

        for batch in mocap_data_loader:
            img_h = batch["img_h"].to(device).float()
            img_w = batch["img_w"].to(device).float()

            focal_length = batch["focal_length"].to(device).float()
            camera_center = torch.hstack((img_w[:, None], img_h[:, None])) / 2

            kpts = np.zeros((batch_size, 49, 3))
            kpts[:, :25, :] = keypoints
            keypoints_tensor = torch.from_numpy(kpts).to(device)

            smpl_poses = smpl.mean_poses[:N_POSES * 3].unsqueeze(0).repeat(batch_size, 1).to(device)  # N*66
            bend_pose = torch.from_numpy(np.load(constants.BEND_POSE_PATH)['pose'][:, :N_POSES * 3]).to(smpl_poses.device)
            smpl_poses[bend_init, 3:] = bend_pose[:, 3:]
            start_time = time.time()
            init_latent = invert_to_latent(net, smpl_poses[:, 3:], latent_size, device, lr=0.5, body_model=smpl)
            end_time = time.time()
            print(f'elapsed time for inversion: {end_time - start_time:.2f}s')
            init_betas = smpl.mean_shape.unsqueeze(0).repeat(batch_size, 1).to(device)  # N*10

            # Convert the camera parameters from the crop camera to the full camera
            if args.init_camera == 'fixed':
                center = batch["center"].to(device).float()
                scale = batch["scale"].to(device).float()
                full_img_shape = torch.stack((img_h, img_w), dim=-1)
                pred_cam_crop = torch.tensor([[0.9, 0, 0]], device=device).repeat(batch_size, 1)
                init_cam_t = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
            else:
                init_joints_3d = smpl(betas=init_betas,
                                      body_pose=smpl_poses[:, 3:],
                                      global_orient=smpl_poses[:, :3], ).joints
                init_cam_t = guess_init(init_joints_3d[:, :25], keypoints_tensor[:, :25], focal_length, part='body')

            init_vertices = smpl(betas=init_betas,
                                 body_pose=smpl_poses[:, 3:],
                                 global_orient=smpl_poses[:, :3],
                                 transl=init_cam_t).vertices

            # be careful: the estimated focal_length should be used here instead of the default constant
            smplify = SMPLify(body_model=smpl, step_size=1.0, batch_size=batch_size, num_iters=150,
                              focal_length=focal_length, args=args, init_latent=init_latent)
            results = smplify(smpl_poses.detach(),
                              init_betas.detach(),
                              init_cam_t.detach(),
                              camera_center,
                              keypoints_tensor)

            new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
            batch_results = mocap_db.eval_EHF(results, batch_ply)
            all_eval_results['pa_mpjpe_body'].extend(batch_results['pa_mpjpe_body'])
            all_eval_results['mpjpe_body'].extend(batch_results['mpjpe_body'])

            if enable_visual:
                # visualize predicted mesh
                pred_output = smpl(betas=new_opt_betas,
                                   body_pose=new_opt_pose[:, 3:],
                                   global_orient=new_opt_pose[:, :3],
                                   transl=new_opt_cam_t)
                pred_vertices = pred_output.vertices
                batch_img_rgb = [img[:, :, ::-1] for img in batch_img]
                renderer = Renderer(focal_length=focal_length[0], img_w=img_w[0], img_h=img_h[0],
                                    faces=smpl.faces,
                                    same_mesh_color=True)
                front_view_list = renderer.render_multiple_front_view(pred_vertices.detach().cpu().numpy(),
                                                                      [img.copy() for img in batch_img_rgb])
                renderer.delete()
                for img_name, front_view in zip(img_names[current_idx: current_idx + batch_size], front_view_list):
                    cv2.imwrite(os.path.join(args.outdir, f"{img_name}_mesh_fit.jpg"), front_view[:, :, ::-1])
                print('img saved in:', args.outdir)

        batch_keypoints, batch_img, batch_ply = [], [], []  # clear for the next batch
        current_idx += batch_size

    print('results on whole dataset:')
    mocap_db.print_eval_result(all_eval_results)

'''
RUN:
python -m experiments.EHF --data_dir=path_to_EHF --prior generator --device cuda:2 --kpts vitpose
'''