"""Sample poses from manifold"""

import argparse
import os
import sys
from functools import partial

from torch import nn
from torch.utils.data import DataLoader

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))

sys.path.insert(0, parent_dir)

import numpy as np
import torch
from configs.config import load_config
# from model_quat import  train_manifold2 as train_manifold
from model.posendf import PoseNDF
from pytorch3d.io import save_obj
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion
from torch.autograd import grad

from experiments.exp_utils import renderer, quat_flip
sys.path.insert(0, '/data3/ljz24/projects/3d/DPoser')
from lib.dataset.body.AMASS import AMASSDataset
from lib.dataset.body import Evaler
from lib.utils.misc import create_mask
from lib.body_model.body_model import BodyModel
from lib.body_model.utils import BodyPartIndices
from lib.body_model.visual import multiple_render
# General config


device = 'cuda:0'

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


class SamplePose(object):
    def __init__(self, posendf, body_model, out_path='./experiment_results/sample_pose', debug=False, device=None,
                 batch_size=1):
        self.debug = debug
        self.device = device
        self.pose_prior = posendf
        self.body_model = body_model
        self.out_path = out_path
        self.betas = torch.zeros((batch_size, 10)).to(device=self.device)  # for visualization
        self.data_loss = nn.MSELoss(reduction='mean')

    @staticmethod
    def visualize(vertices, faces, out_path, device, joints=None, render=False, prefix='out', save_mesh=False):
        # save meshes and rendered results if needed
        os.makedirs(out_path, exist_ok=True)
        if save_mesh:
            os.makedirs(os.path.join(out_path, 'meshes'), exist_ok=True)
            [save_obj(os.path.join(out_path, 'meshes', '{}_{:04}.obj'.format(prefix, i)), vertices[i], faces) for i in
             range(len(vertices))]

        if render:
            renderer(vertices, faces, out_path, device=device, prefix=prefix)

    def project(self, noisy_poses, mask, iterations=100, vis=False, hypo=0):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        batch_size = len(noisy_poses)
        if vis:
            smpl_init = self.body_model(betas=self.betas, pose_body=noisy_poses.view(-1, 63))
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, device=self.device, joints=smpl_init.Jtr,
                           render=True, prefix='init')
        noisy_poses = axis_angle_to_quaternion(noisy_poses.view(batch_size, 21, 3))
        noisy_poses, _ = quat_flip(noisy_poses)
        noisy_poses = torch.nn.functional.normalize(noisy_poses, dim=-1)
        oberservations = noisy_poses.clone()

        noisy_poses.requires_grad = True

        for it in range(iterations):
            # flip and normalize pose before projecting
            net_pred = self.pose_prior(noisy_poses, train=False)
            grad_val = gradient(noisy_poses, net_pred['dist_pred']).reshape(-1, 84)
            noisy_poses = noisy_poses.detach()
            net_pred['dist_pred'] = net_pred['dist_pred'].detach()
            grad_norm = torch.nn.functional.normalize(grad_val, p=2.0, dim=-1)
            noisy_poses = noisy_poses - (net_pred['dist_pred'] * grad_norm).reshape(-1, 21, 4)
            noisy_poses, _ = quat_flip(noisy_poses)
            noisy_poses = torch.nn.functional.normalize(noisy_poses, dim=-1)
            noisy_poses = noisy_poses.detach() * (1.0 - mask) + oberservations * mask
            noisy_poses = noisy_poses.detach()
            noisy_poses.requires_grad = True

            # print(torch.mean(net_pred['dist_pred']))
            # grad = gradient(noisy_poses, net_pred['dist_pred']).reshape(-1, 84)
            # grad_norm = torch.nn.functional.normalize(grad, p=2.0, dim=-1)
            # noisy_poses = noisy_poses - (net_pred['dist_pred']*grad_norm).reshape(-1, 21,4)
            # noisy_poses = torch.nn.functional.normalize(noisy_poses,dim=-1)

        # noisy_poses = noisy_poses.detach() * (1.0 - mask) + oberservations * mask
        clean_poses = quaternion_to_axis_angle(noisy_poses)

        if vis:
            smpl_init = self.body_model(betas=self.betas, pose_body=clean_poses.view(-1, 63))
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, device=self.device, joints=smpl_init.Jtr,
                           render=True, prefix=f'out{hypo}')

        return clean_poses.view(batch_size, -1)

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'data': lambda cst, it: 100 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 1e4 * cst * cst / (1 + it)}
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def optimize(self, observation, mask, lr=2e-5, iterations=2, steps_per_iter=100):
        batchsize, data_dim = observation.shape

        opti_variable = torch.randn(batchsize, data_dim).to(observation.device) * 0.02
        opti_variable.requires_grad = True

        full_data = torch.where(mask, observation, opti_variable)

        optimizer = torch.optim.Adam([opti_variable], lr=lr, betas=(0.9, 0.999))
        weight_dict = self.get_loss_weights()
        loss_dict = dict()

        for it in range(iterations):
            for i in range(steps_per_iter):
                optimizer.zero_grad()

                # convert pose to quaternion and predict distance
                pose_quat = axis_angle_to_quaternion(full_data.view(-1, 21, 3))
                pose_quat, _ = quat_flip(pose_quat)
                pose_quat = torch.nn.functional.normalize(pose_quat, dim=-1)

                dis_val = self.pose_prior(pose_quat, train=False)['dist_pred']
                loss_dict['pose_pr'] = torch.mean(dis_val)
                loss_dict['data'] = self.data_loss(full_data * mask.float(), observation * mask.float())
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                # print(it*steps_per_iter+i, loss_dict['data'], loss_dict['pose_pr'], tot_loss)
                tot_loss.backward()

                optimizer.step()

                full_data = torch.where(mask, observation, opti_variable)

        return full_data


def completion(opt, ckpt, part, hypo_num, out_path=None):
    ### load the model
    net = PoseNDF(opt)
    ckpt = torch.load(ckpt, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device)
    net.eval()

    test_dataset = AMASSDataset(root_path='/data3/ljz24/projects/3d/DPoser/body_data',
                                version='version1', subset='test', sample_interval=50,)
    batch_size = 100
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=False,
                             drop_last=True)
    body_model = BodyModel(bm_path='/data3/ljz24/projects/3d/body_models/smplx/SMPLX_NEUTRAL.npz',
                           num_betas=10,
                           batch_size=batch_size,
                           model_type='smplx').to(device)
    # create Motion denoiser layer
    pose_sampler = SamplePose(net, body_model=body_model, batch_size=batch_size, out_path=out_path, device=device)

    collected_results = []
    collected_dict = {}
    for idx, batch_data in enumerate(test_loader):
        vis = True if idx == 0 else False
        gts = batch_data['body_pose'].to(device, non_blocking=True)
        all_hypos = []
        mask_joints = getattr(BodyPartIndices, part)
        quan_mask = torch.ones((batch_size, 21 * 4), device=device)
        mask_indices = torch.tensor(mask_joints).view(-1, 1) * 4 + torch.arange(4).view(1, -1)
        mask_indices = mask_indices.flatten()
        quan_mask[:, mask_indices] = 0
        mask, observation = create_mask(gts, part=part, observation_type='noise')
        for i in range(hypo_num):
            completion = pose_sampler.optimize(observation, mask)  # [batch_size, 32]
            all_hypos.append(completion)
        all_hypos = torch.stack(all_hypos, dim=1)  # [batch_size, hypo, 32]

        evaler = Evaler(body_model=body_model, part=part)

        eval_results = evaler.multi_eval_bodys_all(all_hypos, gts)  # [batch_size, ]
        collected_results.append(eval_results)

    for single_process_results in collected_results:
        for key, value in single_process_results.items():
            if key not in collected_dict:
                collected_dict[key] = []
            collected_dict[key].extend(value)  # 合并数组

    print(f'results for {hypo_num} evals')
    for key, value in collected_dict.items():
        average_value = np.mean(np.array(value))
        print(f"The average of {key} is {average_value}")


def toy_completion(opt, ckpt, part, hypo_num, view, out_path=None):
    ### load the model
    net = PoseNDF(opt)
    ckpt = torch.load(ckpt, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device)
    net.eval()
    sample_num = 20
    # create Motion denoiser layer
    body_model = BodyModel(bm_path='/data3/ljz24/projects/3d/body_models/smplx/SMPLX_NEUTRAL.npz',
                           num_betas=10,
                           batch_size=sample_num,
                           model_type='smplx').to(device)
    pose_sampler = SamplePose(net, body_model=body_model, batch_size=sample_num, out_path=out_path, device=device)

    file_path = '/data3/ljz24/projects/3d/DPoser/examples/toy_body_data.npz'
    data = np.load(file_path, allow_pickle=True)
    body_poses = data['pose_samples'][:sample_num]
    print(f'loaded axis pose data {body_poses.shape} from {file_path}')
    gts = torch.from_numpy(body_poses).to(device)
    mask, observation = create_mask(gts, part=part, observation_type='mean')

    mask_joints = getattr(BodyPartIndices, part)
    quan_mask = torch.ones((sample_num, 21 * 4), device=device)
    mask_indices = torch.tensor(mask_joints).view(-1, 1) * 4 + torch.arange(4).view(1, -1)
    mask_indices = mask_indices.flatten()
    quan_mask[:, mask_indices] = 0

    all_hypos = []
    for i in range(hypo_num):
        completion = pose_sampler.optimize(observation, mask)  # [batch_size, 32]
        # completion = pose_sampler.project(observation, quan_mask.reshape(sample_num, 21, 4),
        #                                   iterations=100, hypo=i)  # [batch_size, 32]
        all_hypos.append(completion)
    all_hypos = torch.stack(all_hypos)  # [hypo, batch_size, 32]

    evaler = Evaler(body_model=body_model, part=part)
    eval_results = evaler.multi_eval_bodys_all(all_hypos.transpose(0, 1), gts)  # [batch_size, ]
    evaler.print_eval_result_all(eval_results)

    bg_img = np.ones([512, 384, 3]) * 255  # background canvas
    focal = [1500, 1500]
    princpt = [200, 192]
    save_renders = partial(multiple_render, bg_img=bg_img, focal=focal, princpt=princpt, device=device)

    save_renders(gts, None, body_model, out_path, 'sample{}_original.png', convert=False, faster=False,
                 view=view)
    print(f'Original samples under {out_path}')
    save_renders(observation, None, body_model, out_path, 'sample{}_masked.png', convert=False, faster=False,
                 view=view)
    print(f'Masked samples under {out_path}')

    for i in range(hypo_num):
        save_renders(all_hypos[i], None, body_model, out_path, 'sample{}_completion' + str(i) + '.png',
                     convert=False, faster=False, view=view)
    print(f'Completion samples under {out_path}')


if __name__ == '__main__':
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(
        description='Pose completion using PoseNDF.'
    )
    parser.add_argument('--config', '-c', default='./checkpoints/config.yaml', type=str, help='Path to config file.')
    parser.add_argument('--ckpt-path', '-ckpt', default='./checkpoints/checkpoint_v2.tar', type=str,
                        help='Path to pretrained model.')
    parser.add_argument('--noisy_pose', '-np', default=None, type=str, help='Path to noisy motion file')
    parser.add_argument('--outpath_folder', '-out', default='./vposer_completion/legs', type=str,
                        help='Path to output')
    parser.add_argument('--view', default='front', type=str, help='view direction')
    parser.add_argument('--part', default='legs', type=str, help='mask part')
    args = parser.parse_args()

    opt = load_config(args.config)

    # completion(opt, args.ckpt_path, part=args.part, hypo_num=10, out_path=args.outpath_folder)
    toy_completion(opt, args.ckpt_path, part=args.part, view=args.view, hypo_num=10, out_path=args.outpath_folder)
    '''
    RUN:
    python -m experiments.pose_completion --part legs --outpath_folder ./PoseNDF_eccv_exp/completion_results/right_arm --view right_half
    '''

