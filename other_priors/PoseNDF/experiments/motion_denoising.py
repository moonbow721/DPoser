import argparse
import os
import random
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))

sys.path.insert(0, parent_dir)

import cv2
import numpy as np
import torch
from configs.config import load_config
# from model_quat import  train_manifold2 as train_manifold
from model.posendf import PoseNDF
from pytorch3d.io import save_obj
from pytorch3d.transforms import axis_angle_to_quaternion
from tqdm import tqdm

from experiments.body_model import BodyModel
from experiments.exp_utils import renderer, quat_flip

sys.path.insert(0, '/data3/ljz24/projects/3d/DPoser')
from lib.body_model.visual import render_mesh

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]
POSE_DIM = 63  # 69 for smpl
SMPL_CLASS = 'smplx'
# General config

class MotionDenoise(object):
    def __init__(self, posendf, body_model, out_path='./experiment_results/motion_denoise', debug=False,
                 device='cuda:0', batch_size=1, gender='male', render=False, pose_pr_factor=7):
        self.debug = debug
        self.device = device
        self.pose_prior = posendf
        self.body_model = body_model
        self.out_path = out_path
        self.render = render
        self.betas = torch.zeros((batch_size, 10)).to(device=self.device)
        self.poses = torch.randn((batch_size, POSE_DIM)).to(device=self.device) * 0.01
        self.pose_pr_factor = pose_pr_factor

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'temp': lambda cst, it: 10. ** 1 * cst * (1 + it),
                       'data': lambda cst, it: 10. ** 2 * cst / ((1 + it * it)),
                       'pose_pr': lambda cst, it: 10. ** self.pose_pr_factor * cst * cst * (1 + it)
                       }
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    @staticmethod
    def visualize(vertices, faces, out_path, device, joints=None, render=False, prefix='out', save_mesh=False):
        # save meshes and rendered results if needed
        if save_mesh:
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(os.path.join(out_path, 'meshes'), exist_ok=True)
            [save_obj(os.path.join(out_path, 'meshes', '{}_{:04}.obj'.format(prefix, i)), vertices[i], faces) for i in
             range(len(vertices))]

        if render:
            os.makedirs(os.path.join(out_path, 'renders'), exist_ok=True)
            vertices = vertices.detach().cpu()
            faces = faces.cpu()
            for i in range(len(vertices)):
                rendered_img = render_mesh(bg_img, vertices[i], faces, {'focal': focal, 'princpt': princpt},
                                           view='front')
                cv2.imwrite(os.path.join(out_path, 'renders', '{}_{:04}.png'.format(prefix, i)), rendered_img)

    def optimize(self, joints3d, gt_poses=None, iterations=5, steps_per_iter=50):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        smpl_init = self.body_model(betas=self.betas, pose_body=self.poses)
        smpl_gt = self.body_model(betas=self.betas, pose_body=gt_poses)
        self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, device=self.device, joints=smpl_init.Jtr,
                       render=self.render, prefix='init')
        self.visualize(smpl_gt.vertices, smpl_gt.faces, self.out_path, device=self.device, joints=smpl_init.Jtr,
                       render=self.render, prefix='gt')

        joint_error = joints3d - smpl_gt.Jtr[:, :22]
        joint_error = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2))) * 100.
        print('before denoising:{:0.8f} cm'.format(joint_error))

        init_joints = joints3d.detach()

        # Optimizer
        smpl_init.body_pose.requires_grad = True

        optimizer = torch.optim.Adam([smpl_init.body_pose], 0.03, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL poses')
            for i in loop:
                optimizer.zero_grad()
                loss_dict = dict()
                # convert pose to quaternion and  predict distance
                pose_quat = axis_angle_to_quaternion(smpl_init.body_pose.view(-1, POSE_DIM//3, 3)[:, :21])
                pose_quat, _ = quat_flip(pose_quat)
                pose_quat = torch.nn.functional.normalize(pose_quat, dim=-1)

                dis_val = self.pose_prior(pose_quat, train=False)['dist_pred']
                loss_dict['pose_pr'] = torch.mean(dis_val)

                # calculate temporal loss between mesh vertices
                smpl_init = self.body_model(betas=smpl_init.betas, pose_body=smpl_init.body_pose)
                temp_term = smpl_init.vertices[:-1] - smpl_init.vertices[1:]
                loss_dict['temp'] = torch.mean(torch.sqrt(torch.sum(temp_term * temp_term, dim=2)))

                # calculate data term from inital noisy pose
                data_term = smpl_init.Jtr[:, :22] - init_joints
                data_term = torch.mean(torch.sqrt(torch.sum(data_term * data_term, dim=2)))
                if data_term > 0:  # for nans
                    loss_dict['data'] = data_term

                # only for check
                joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
                joint_error = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2))) * 100.

                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Step: {} Iter: {}'.format(it, i)
                l_str += ' v2v : {:0.8f}'.format(joint_error)
                l_str += ' total : {:0.8f}'.format(tot_loss)
                for k in loss_dict:
                    l_str += ', {}: {:0.8f}'.format(k, loss_dict[k].mean().item())
                    loop.set_description(l_str)

        smpl_init = self.body_model(betas=self.betas, pose_body=smpl_init.body_pose)
        # self.visualize(smpl_init.vertices, smpl_init.faces, self.out_path, render=True, prefix='out',
        #                device=self.device)
        joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
        joint_error = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2))) * 100.
        vert_error = smpl_init.vertices - smpl_gt.vertices
        vert_error = torch.mean(torch.sqrt(torch.sum(vert_error * vert_error, dim=2))) * 100.
        print('after denoising:{:0.8f} cm'.format(joint_error))
        return joint_error.detach().cpu().numpy(), vert_error.detach().cpu().numpy(), smpl_init.body_pose.detach().cpu().numpy(), smpl_init.betas.detach().cpu().numpy()


def main(opt, ckpt, gt_data=None, out_path=None, render=False, std=0.04):
    ### load the model
    net = PoseNDF(opt)
    device = 'cuda:0'
    ckpt = torch.load(ckpt, map_location='cpu')['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)

    motion_data_gt = np.load(gt_data)['pose_body']
    batch_size = len(motion_data_gt)
    pose_body = torch.from_numpy(motion_data_gt.astype(np.float32)).to(device)
    gt_poses = torch.zeros((batch_size, POSE_DIM)).to(device)  # 69 for smpl
    gt_poses[:, :63] = pose_body

    #  load body model
    bm_dir_path = f'../body_models/{SMPL_CLASS}'
    body_model = BodyModel(bm_path=bm_dir_path, model_type=SMPL_CLASS, batch_size=batch_size, num_betas=10).to(device=device)

    # generate noise on joints
    joints3d = body_model(pose_body=gt_poses).Jtr[:, :22]
    noisy_joints3d = joints3d + std * torch.randn(*joints3d.shape, device=joints3d.device)

    if std == 0.02:
        kwargs = {'iterations': 3, 'steps_per_iter': 50,}
        pose_pr_factor = 6
    elif std == 0.04:
        kwargs = {'iterations': 3, 'steps_per_iter': 60}
        pose_pr_factor = 7
    elif std == 0.1:
        kwargs = {'iterations': 3, 'steps_per_iter': 80,}
        pose_pr_factor = 8
    else:
        raise NotImplementedError()
    # create Motion denoiser layer
    motion_denoiser = MotionDenoise(net, body_model=body_model, batch_size=len(noisy_joints3d), out_path=out_path,
                                    render=render, pose_pr_factor=pose_pr_factor)
    j2j_err, v2v_err, pose, betas = motion_denoiser.optimize(noisy_joints3d, gt_poses, **kwargs)

    # np.savez(os.path.join(out_path, seq + '.npz'), v2v_error=v2v_err, pose_body=pose, betas=betas)
    return j2j_err, v2v_err


if __name__ == '__main__':
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(
        description='Motion denoising using PoseNDF.'
    )
    parser.add_argument('--config', '-c', default='./checkpoints/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--ckpt_path', '-ckpt', default='./checkpoints/checkpoint_epoch_best.tar', type=str,
                        help='Path to pretrained model.')
    parser.add_argument('--outpath_folder', '-out', default='./PoseNDF_exp/fitting3d_results', type=str,
                        help='Path to output')
    parser.add_argument('--noise-std', type=float, default=0.04, help='control added noise')
    parser.add_argument('--dataset', type=str, default='AMASS')
    args = parser.parse_args()

    opt = load_config(args.config)

    gt_path = '/data3/ljz24/projects/3d/DPoser/examples/Gestures_3_poses_batch005.npz'
    j2j_err, v2v_err = main(opt, args.ckpt_path, gt_data=gt_path, out_path=args.outpath_folder, std=args.noise_std)
    print(j2j_err, v2v_err)

    '''
    RUN:
    python -m experiments.motion_denoising --noise-std 0.04 
    '''