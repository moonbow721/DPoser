import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm

from lib.body_model import constants
from lib.body_model.visual import save_obj, render_mesh, faster_render, vis_body_skeletons
from lib.algorithms.advanced import sde_lib
from lib.algorithms.advanced import utils as mutils
from lib.algorithms.advanced.model import create_model
from lib.body_model.body_model import BodyModel
from lib.utils.generic import load_model
from lib.utils.misc import lerp, create_joint_mask, create_stable_mask, gaussian_smoothing
from lib.utils.transforms import rot6d_to_axis_angle
from lib.utils.generic import find_npz_files


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

from lib.dataset.body import N_POSES
from lib.dataset.utils import Posenormalizer

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='visualize the save files and demo on toy data')

    parser.add_argument('--ckpt-path', type=str,
                        default='./pretrained_models/amass/BaseMLP/epoch=36-step=150000-val_mpjpe=38.17.ckpt',
                        help='load trained diffusion model for DPoser')
    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX')
    parser.add_argument('--dataset-folder', type=str,
                        default='../data/human/Bodydataset/amass_processed',
                        help='the folder includes necessary normalizing parameters')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--sample', type=int, default=1, help='reduce test samples')

    parser.add_argument('--outpath-folder', type=str, default='./output/body/test_results/partial_motion_denoise')
    parser.add_argument('--part', type=str, default='legs', help='the masked part')
    parser.add_argument('--view', type=str, default='front', help='render view')
    parser.add_argument('--noise-std', type=float, default=0.0, help='control added noise')
    parser.add_argument('--time-strategy', type=str, default='3', choices=['1', '2', '3'],
                        help='random, fix, truncated annealing')
    parser.add_argument('--device', type=str, default='cuda:0')

    # data preparation
    parser.add_argument('--file-path', type=str, help='use toy data to run')
    parser.add_argument('--dataset', type=str, default='AMASS', choices=['AMASS', 'HPS'])

    args = parser.parse_args(argv[1:])

    return args


class MotionDenoise(object):
    def __init__(self, config, args, diffusion_model, body_model, sds_weight=1.0,
                 out_path=None, debug=False, view='front', batch_size=1, pose_init=None):
        self.args = args
        self.debug = debug
        self.view = view
        self.device = args.device
        self.body_model = body_model
        self.sds_weight = sds_weight
        self.out_path = out_path  # only needed for visualization
        self.batch_size = batch_size
        self.betas = torch.zeros((batch_size, 10), device=self.device)
        self.poses = torch.randn((batch_size, 63), device=self.device) * 0.01 if pose_init is None else pose_init
        self.Normalizer = Posenormalizer(
            data_path=f'{args.dataset_folder}/{args.version}/train',
            normalize=config.data.normalize,
            min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)

        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        self.sde = sde
        self.score_fn = mutils.get_score_fn(sde, diffusion_model, train=False, continuous=config.training.continuous)
        self.rsde = sde.reverse(self.score_fn, False)
        # L2 loss
        self.loss_fn = nn.MSELoss(reduction='none')

    def one_step_denoise(self, x_t, t):
        drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x_t, t, guide=True)
        x_0_hat = (x_t + sigma_2[:, None] * score) / alpha
        SNR = alpha / torch.sqrt(sigma_2)[:, None]

        return x_0_hat.detach(), SNR

    def multi_step_denoise(self, x_t, t, t_end, N=10):
        time_traj = lerp(t, t_end, N + 1)
        x_current = x_t

        for i in range(N):
            t_current = time_traj[i]
            t_before = time_traj[i + 1]
            alpha_current, sigma_current = self.sde.return_alpha_sigma(t_current)
            alpha_before, sigma_before = self.sde.return_alpha_sigma(t_before)
            score = self.score_fn(x_current, t_current, condition=None, mask=None)
            score = -score * sigma_current[:, None]  # score to noise prediction
            x_current = alpha_before / alpha_current * (x_current - sigma_current[:, None] * score) + sigma_before[
                                                                                                      :,
                                                                                                      None] * score
        alpha, sigma = self.sde.return_alpha_sigma(time_traj[0])
        SNR = alpha / sigma[:, None]
        return x_current.detach(), SNR

    # In our experiments, we found multi-step denoise will lead to worse results.
    def DPoser_loss(self, x_0, vec_t, weighted=True, multi_denoise=False):
        # x_0: [B, j*6], vec_t: [B], quan_t: [1]
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #

        if multi_denoise:
            denoise_data, SNR = self.multi_step_denoise(perturbed_data, vec_t, t_end=vec_t / (2 * 10), N=10)
        else:
            denoise_data, SNR = self.one_step_denoise(perturbed_data, vec_t)

        if weighted:
            weight = 0.5 * torch.sqrt(1 + SNR)
        else:
            weight = 0.5

        sds_loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / self.batch_size

        return sds_loss

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'temp': lambda cst, it: 20. * cst,
                       'data': lambda cst, it: 10. ** 2 * cst,
                       'dposer': lambda cst, it: 1.0 * cst * (1 + it) * self.sds_weight
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
    def visualize(vertices, faces, out_path, render=False, prefix='out', view='front', save_mesh=False, faster=False, device=None):
        # save meshes and rendered results if needed
        os.makedirs(out_path, exist_ok=True)
        if save_mesh:
            vertices = vertices.detach().cpu()
            faces = faces.cpu()
            os.makedirs(os.path.join(out_path, 'meshes'), exist_ok=True)
            [save_obj(vertices[i], faces, os.path.join(out_path, 'meshes', '{}_{:04}.obj'.format(prefix, i))) for i in
             range(len(vertices))]

        if render:
            os.makedirs(os.path.join(out_path, 'renders'), exist_ok=True)
            if faster:
                assert device is not None
                target_path = os.path.join(out_path, 'renders')
                faster_render(vertices, faces, target_path, prefix + '_{:04}.jpg', device)
            else:
                vertices = vertices.detach().cpu()
                faces = faces.cpu()
                for i in range(len(vertices)):
                    rendered_img = render_mesh(bg_img, vertices[i], faces, {'focal': focal, 'princpt': princpt},
                                               view=view)
                    cv2.imwrite(os.path.join(out_path, 'renders', '{}_{:04}.png'.format(prefix, i)), rendered_img)

    def optimize(self, joints3d, mask, gt_poses=None, time_strategy='1', post_smooth=False,
                 t_max=0.2, t_min=1e-3, t_fixed=0.1, iterations=5, steps_per_iter=50, verbose=False, vis=False):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        smpl_init = self.body_model(betas=self.betas, body_pose=self.poses)
        smpl_gt = self.body_model(betas=self.betas, body_pose=gt_poses)
        if vis:
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, render=True, prefix='init', faster=True,
                           device=self.device)
            visible_mask = mask[0, :, :1].detach().cpu().numpy().astype(np.float32)
            vis_body_skeletons(joints3d.detach().cpu().numpy(), kpt_3d_vis=visible_mask, output_path=os.path.join(self.out_path, 'renders'))
            self.visualize(smpl_gt.v, smpl_gt.f, self.out_path, render=True, prefix='gt', view=self.view, device=self.device)

        joint_error = joints3d - smpl_gt.Jtr[:, :22]
        init_MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 100.
        if verbose:
            print('before denoising:{:0.8f} cm'.format(init_MPJPE.mean()))

        init_joints = joints3d.detach()

        # Optimizer
        mask.requires_grad = False
        smpl_init.body_pose.requires_grad = True
        optimizer = torch.optim.Adam([smpl_init.body_pose], 0.03, betas=(0.9, 0.999), eps=1e-6)
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        stable_mask = create_stable_mask(mask, eps=1e-20)
        total_steps = iterations * steps_per_iter
        for it in range(iterations):
            if verbose:
                loop = tqdm(range(steps_per_iter))
                loop.set_description('Optimizing SMPL poses')
            else:
                loop = range(steps_per_iter)

            for i in loop:
                step = it * steps_per_iter + i
                optimizer.zero_grad()
                loss_dict = dict()

                '''   *************      DPoser loss ***********         '''
                poses = self.Normalizer.offline_normalize(smpl_init.body_pose, from_axis=True)
                eps = 1e-3
                if time_strategy == '1':
                    t = eps + torch.rand(1) * (self.sde.T - eps)
                elif time_strategy == '2':
                    t = torch.tensor(t_fixed)
                elif time_strategy == '3':
                    t = t_min + torch.tensor(total_steps - step - 1) / total_steps * (t_max - t_min)
                else:
                    raise NotImplementedError
                vec_t = torch.ones(self.batch_size, device=self.device) * t
                loss_dict['dposer'] = self.DPoser_loss(poses, vec_t)
                '''   ***********      DPoser loss   ************       '''

                # calculate temporal loss between mesh vertices
                smpl_init = self.body_model(betas=smpl_init.betas, body_pose=smpl_init.body_pose)
                temp_term = smpl_init.v[:-1] - smpl_init.v[1:]
                if it >= 2:
                    loss_dict['temp'] = torch.mean(torch.sqrt(torch.sum(temp_term * temp_term, dim=2)))

                # calculate data term from inital noisy pose
                data_term = (smpl_init.Jtr[:, :22] - init_joints) * stable_mask
                data_term = torch.mean(torch.sqrt(torch.sum(data_term * data_term, dim=2)))
                # print(data_term)
                if data_term > 0:  # for nans
                    loss_dict['data'] = data_term

                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()

                optimizer.step()

                if verbose:
                    # only for check
                    joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
                    joint_error = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2))) * 100.

                    l_str = 'Step: {} Iter: {}'.format(it, i)
                    l_str += ' j2j : {:0.8f}'.format(joint_error)
                    l_str += ' total : {:0.8f}'.format(tot_loss)
                    for k in loss_dict:
                        l_str += ', {}: {:0.8f}'.format(k, loss_dict[k].mean().item())
                    loop.set_description(l_str)

        if post_smooth:
            smooth_pose = gaussian_smoothing(smpl_init.body_pose.clone(), window_size=5, sigma=3)
            idx = [0, 1, -1, -2]
            smooth_pose[idx] = smpl_init.body_pose[idx]
            smpl_init = self.body_model(betas=self.betas, body_pose=smooth_pose)
        else:
            smpl_init = self.body_model(betas=self.betas, body_pose=smpl_init.body_pose)

        joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
        vert_error = smpl_init.v - smpl_gt.v
        MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 100.  # remain batch dim
        MPVPE = torch.mean(torch.sqrt(torch.sum(vert_error * vert_error, dim=2)), dim=1) * 100.
        # visible joints, all frames share the same mask
        joint_mask = mask[0, :, 0]
        visible_joint_error = joint_error[:, joint_mask]
        vis_MPJPE = torch.mean(torch.sqrt(torch.sum(visible_joint_error * visible_joint_error, dim=2)), dim=1) * 100.
        invisible_joint_error = joint_error[:, ~joint_mask]
        inv_MPJPE = torch.mean(torch.sqrt(torch.sum(invisible_joint_error * invisible_joint_error, dim=2)),
                               dim=1) * 100.
        results_dict = {'init_MPJPE': init_MPJPE.detach().cpu().numpy(), 'MPJPE': MPJPE.detach().cpu().numpy(),
                        'MPVPE': MPVPE.detach().cpu().numpy(), 'vis_MPJPE': vis_MPJPE.detach().cpu().numpy(),
                        'inv_MPJPE': inv_MPJPE.detach().cpu().numpy(),}
        if vis:
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, render=True,
                           prefix='out', view=self.view, device=self.device)

        if verbose:
            print('after denoising:{:0.8f} cm'.format(MPJPE.mean()))

        return results_dict


def denoise(config, args, model, gt_file, out_path, std=0.04, part='arms', verbose=False):
    motion_data_gt = np.load(gt_file)['pose_body']
    if args.dataset == 'HPS':
        motion_data_gt = motion_data_gt[:, 3:N_POSES * 3 + 3]  # skip root orient, abandon hands(SMPL->SMPLX)
    batch_size = len(motion_data_gt)
    gt_poses = torch.from_numpy(motion_data_gt.astype(np.float32)).to(args.device)  # [batchsize, 63]

    #  load body model
    body_model = BodyModel(bm_path=args.bodymodel_path, model_type='smplx', batch_size=batch_size, num_betas=10).to(
        args.device)

    # generate noise on joints
    assert std == 0.0, "std should be 0.0 for completion"
    joints3d = body_model(body_pose=gt_poses).Jtr[:, :22]
    noisy_joints3d = joints3d + std * torch.randn(*joints3d.shape, device=joints3d.device)
    mask, partial_joints3d = create_joint_mask(noisy_joints3d, part=part, )  # [batchsize, 22, 3]

    if part == 'legs':
        kwargs = {'iterations': 3, 'steps_per_iter': 80, 't_max': 0.2, 't_min': 0.01, 't_fixed': 0.05}
        sds_weight = 0.4
    elif part == 'left_arm':
        kwargs = {'iterations': 3, 'steps_per_iter': 80, 't_max': 0.3, 't_min': 0.01, 't_fixed': 0.05}
        sds_weight = 0.2
    elif part == 'right_body':
        kwargs = {'iterations': 3, 'steps_per_iter': 80, 't_max': 0.3, 't_min': 0.01, 't_fixed': 0.05}
        sds_weight = 0.3
    else:
        raise NotImplementedError(f"part {part} not implemented")

    # create Motion denoiser layer
    motion_denoiser = MotionDenoise(config, args, model, body_model=body_model,
                                    sds_weight=sds_weight, batch_size=batch_size,
                                    out_path=out_path, view=args.view, pose_init=None)

    if args.file_path is not None:  # visualization for toy data
        verbose = True
        kwargs['vis'] = True
    try:
        batch_results = motion_denoiser.optimize(partial_joints3d, mask, gt_poses, args.time_strategy,
                                                 verbose=verbose, post_smooth=True, **kwargs)
    except Exception as e:
        print(f"Error in {gt_file}: {e}")
        return None

    # check nan in results
    for key, value in batch_results.items():
        if np.isnan(value).any():
            print(f"nan in {key} for {gt_file}")
            batch_results = None
            break
    return batch_results


def main(args):
    torch.manual_seed(42)
    config = FLAGS.config

    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    model = create_model(config.model, N_POSES, POSE_DIM, )
    model.to(args.device)
    model.eval()
    load_model(model, config, args.ckpt_path, args.device, is_ema=True)

    if args.file_path is not None:
        os.makedirs(args.outpath_folder, exist_ok=True)
        batch_results = denoise(config, args, model, args.file_path, args.outpath_folder, std=args.noise_std,
                                part=args.part)
        for key, value in batch_results.items():
            average_value = np.mean(np.array(value))
            print(f"The average of {key} is {average_value}")
    else:  # run for whole test set
        out_path = None  # No visualization to save time
        if args.dataset == 'AMASS':
            data_dir = args.data_dir
            seqs = sorted(find_npz_files(data_dir))
        else:
            raise NotImplementedError
        num_to_select = len(seqs) // args.sample  # Integer division to get 1/10 of the total
        random.seed(42)
        seqs = sorted(random.sample(seqs, num_to_select))
        print('Sample: {}, Test dataset consists of {} sequences'.format(args.sample, len(seqs)))

        # Initialize a dictionary to store all batch results
        all_batch_results = {}
        # logging
        csv_file_path = os.path.join(args.outpath_folder, f'{args.dataset}_results_logging.csv')
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Sequence', 'init_MPJPE', 'MPJPE', 'MPVPE', 'vis_MPJPE', 'inv_MPJPE'])

        for seq in seqs:
            gt_path = os.path.join(data_dir, seq)
            batch_results = denoise(config, args, model, gt_path, out_path,
                                    std=args.noise_std, part=args.part, verbose=True)
            if batch_results is None:
                continue

            # logging into csv
            with open(csv_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                avg_values = [np.mean(batch_results[key]) for key in ['init_MPJPE', 'MPJPE', 'MPVPE']]
                csv_writer.writerow([seq] + avg_values)

            # Aggregate the batch results into the overall results
            for key, value in batch_results.items():
                if key in all_batch_results:
                    all_batch_results[key] = np.concatenate([all_batch_results[key], value])
                else:
                    all_batch_results[key] = value

        # Calculate the average of all batch results
        for key, value in all_batch_results.items():
            average_value = np.mean(value)
            print(f"The average of {key} is {average_value}")


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)
