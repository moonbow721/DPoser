import csv
import os

import cv2
import math
import numpy as np
import torch
import torch.nn as nn
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from tqdm import tqdm

from lib.body_model.visual import save_obj, render_mesh, faster_render, vis_skeletons
from lib.algorithms.advanced import sde_lib, sampling
from lib.algorithms.advanced import utils as mutils
from lib.algorithms.advanced.model import ScoreModelFC
from lib.algorithms.ema import ExponentialMovingAverage
from lib.body_model.body_model import BodyModel
from lib.utils.misc import linear_interpolation, gaussian_smoothing
from lib.utils.motion_video import seq_to_video

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

from lib.dataset.AMASS import Posenormalizer, N_POSES

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='motion denosing (3D noisy joints -> clean poses)')

    parser.add_argument('--dataset-folder', type=str, default='../data/AMASS/amass_processed',
                        help='the folder includes necessary normalizing parameters')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/axis-zscore-400k.pth')
    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX')
    parser.add_argument('--outpath-folder', type=str, default='./output/test_results/motion_denoise')
    parser.add_argument('--noise-std', type=float, default=0.04, help='control added noise scales')

    parser.add_argument('--time-strategy', type=str, default='3', choices=['1', '2', '3'],
                        help='random, fix, truncated annealing')
    parser.add_argument('--device', type=str, default='cuda:0')

    # data preparation
    parser.add_argument('--file-path', type=str, help='use toy data to run')
    parser.add_argument('--data-dir', type=str, default='../humor/out/amass_joints_noisy_fitting/results_out',
                        help='the whole AMASS testset, (output from HuMoR)')
    parser.add_argument('--dataset', type=str, default='AMASS')

    args = parser.parse_args(argv[1:])

    return args


class MotionDenoise(object):
    def __init__(self, config, args, diffusion_model, body_model, sde_N=1000, dposer_weight=1.0,
                 out_path=None, debug=False, batch_size=1):
        self.args = args
        self.debug = debug
        self.device = args.device
        self.body_model = body_model
        self.dposer_weight = dposer_weight
        self.out_path = out_path    # only needed for visualization
        self.batch_size = batch_size
        self.betas = torch.zeros((batch_size, 10), device=self.device)
        self.poses = torch.randn((batch_size, 63), device=self.device) * 0.01
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
        sde.N = sde_N  # discrete sampling steps

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
        time_traj = linear_interpolation(t, t_end, N + 1)
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
    def DPoser_loss(self, x_0, vec_t, quan_t, weighted=False, multi_denoise=False):
        # x_0: [B, j*6], vec_t: [B], quan_t: [1]
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #

        if multi_denoise:
            denoise_data, SNR = self.multi_step_denoise(perturbed_data, vec_t, t_end=vec_t / (2 * 10), N=10)
        else:
            denoise_data, SNR = self.one_step_denoise(perturbed_data, vec_t)

        if weighted:
            weight = 0.5 * torch.sqrt(1+SNR)
        else:
            weight = 0.5

        dposer_loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / self.batch_size

        return dposer_loss

    def RED_Diff(self, x_0, vec_t, quan_t):
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #
        _, _, alpha, sigma_2, score = self.rsde.sde(perturbed_data, vec_t, guide=True)
        score = -score * std[:, None]   # score to noise prediction
        inverse_SNR = torch.sqrt(sigma_2) / alpha[:, 0]
        weight = inverse_SNR
        guidance = torch.mean(weight * torch.einsum('ij,ij->i', (score - z).detach(), x_0))
        return guidance

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'temp': lambda cst, it: 10. ** 1 * cst * (1 + it),
                       'data': lambda cst, it: 10. ** 2 * cst / (1 + it * it),
                       'dposer': lambda cst, it: 10. ** -1 * cst * (1 + it) * self.dposer_weight
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
    def visualize(vertices, faces, out_path, render=False, prefix='out', save_mesh=False, faster=False, device=None):
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
                                               view='front')
                    cv2.imwrite(os.path.join(out_path, 'renders', '{}_{:04}.png'.format(prefix, i)), rendered_img)

    def optimize(self, joints3d, gt_poses=None, time_strategy='1',
                 sample_trun=2.0, sample_time=990, iterations=5, steps_per_iter=50, verbose=False, vis=False):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        smpl_init = self.body_model(betas=self.betas, pose_body=self.poses)
        smpl_gt = self.body_model(betas=self.betas, pose_body=gt_poses)
        if vis:
            vis_skeletons(joints3d.detach().cpu().numpy(), os.path.join(self.out_path, 'renders'))
            print('skeleton figures saved')
            self.visualize(smpl_gt.v, smpl_gt.f, self.out_path, render=True, prefix='gt', device=self.device)

        joint_error = joints3d - smpl_gt.Jtr[:, :22]
        init_MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 100.
        if verbose:
            print('before denoising:{:0.8f} cm'.format(init_MPJPE.mean()))

        init_joints = joints3d.detach()

        # Optimizer
        smpl_init.pose_body.requires_grad = True
        optimizer = torch.optim.Adam([smpl_init.pose_body], 0.03, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        eps = 1e-3
        timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=self.device)

        total_steps = iterations*steps_per_iter
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
                poses = self.Normalizer.offline_normalize(smpl_init.pose_body, from_axis=True)

                if time_strategy == '1':  # not recommend
                    quan_t = torch.randint(self.sde.N, [1])
                elif time_strategy == '2':
                    quan_t = torch.tensor(sample_time)
                elif time_strategy == '3':
                    quan_t = self.sde.N - math.floor(torch.tensor(total_steps - step - 1) * (self.sde.N / (sample_trun * total_steps))) - 2
                else:
                    raise NotImplementedError('unsupported time sampling strategy')

                t = timesteps[quan_t]
                vec_t = torch.ones(self.batch_size, device=self.device) * t
                loss_dict['dposer'] = self.DPoser_loss(poses, vec_t, quan_t)
                '''   ***********      DPoser loss   ************       '''

                # calculate temporal loss between mesh vertices
                smpl_init = self.body_model(betas=smpl_init.betas, pose_body=smpl_init.pose_body)
                temp_term = smpl_init.v[:-1] - smpl_init.v[1:]
                loss_dict['temp'] = torch.mean(torch.sqrt(torch.sum(temp_term * temp_term, dim=2)))

                # calculate data term from inital noisy pose
                data_term = smpl_init.Jtr[:, :22] - init_joints
                data_term = torch.mean(torch.sqrt(torch.sum(data_term * data_term, dim=2)))
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

        # create final results, the smoothing can be used to create consistent demo videos
        # Note that we do not use the smoothing for evaluation in our paper
        smooth_pose = gaussian_smoothing(smpl_init.pose_body, window_size=3, sigma=2)
        idx = [0, -1]
        smooth_pose[idx] = smpl_init.pose_body[idx]
        smpl_init = self.body_model(betas=self.betas, pose_body=smooth_pose)
        if vis:
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, render=True, prefix='out', device=self.device)
            seq_to_video(os.path.join(self.out_path, 'renders'), os.path.join(self.out_path, 'merges'),
                         video_path=os.path.join(self.out_path, 'motion.mp4'))

        joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
        vert_error = smpl_init.v - smpl_gt.v
        MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 100.   # remain batch dim
        MPVPE = torch.mean(torch.sqrt(torch.sum(vert_error * vert_error, dim=2)), dim=1) * 100.
        if verbose:
            print('after denoising:{:0.8f} cm'.format(MPJPE.mean()))
        results_dict = {'init_MPJPE': init_MPJPE.detach().cpu().numpy(), 'MPJPE': MPJPE.detach().cpu().numpy(),
                        'MPVPE': MPVPE.detach().cpu().numpy()}
        return results_dict


def denoise(config, args, model, gt_file, out_path, std=0.04, verbose=False):
    motion_data_gt = np.load(gt_file)['pose_body']
    batch_size = len(motion_data_gt)
    gt_poses = torch.from_numpy(motion_data_gt.astype(np.float32)).to(args.device)  # [batchsize, 63]

    #  load body model
    body_model = BodyModel(bm_path=args.bodymodel_path, model_type='smplx', batch_size=batch_size, num_betas=10).to(
        args.device)

    # generate noise on joints
    std = std
    joints3d = body_model(pose_body=gt_poses).Jtr[:, :22]
    noisy_joints3d = joints3d + std * torch.randn(*joints3d.shape, device=joints3d.device)

    if args.time_strategy in ['1']:
        sde_N = 500
        dposer_weight = 1e-1
    else:
        sde_N = 500
        dposer_weight = 1.0  # If you try to reduce 'sample_trun' or 'sample_time', reduce weight too for converge.
    # create Motion denoiser layer
    motion_denoiser = MotionDenoise(config, args, model, sde_N=sde_N, body_model=body_model,
                                    dposer_weight=dposer_weight,    # For axis setting 1e-1,
                                    batch_size=batch_size,
                                    out_path=out_path)

    if std == 0.02:
        kwargs = {'iterations': 3, 'steps_per_iter': 40, 'sample_trun': 10.0, 'sample_time': 495}
    elif std == 0.04:
        kwargs = {'iterations': 3, 'steps_per_iter': 60, 'sample_trun': 4.0, 'sample_time': 490}
    elif std == 0.1:
        kwargs = {'iterations': 3, 'steps_per_iter': 80, 'sample_trun': 3.0, 'sample_time': 480}
    else:
        raise NotImplementedError()

    if args.file_path is not None:  # visualization for toy data
        verbose = True
        kwargs['vis'] = True

    batch_results = motion_denoiser.optimize(noisy_joints3d, gt_poses, args.time_strategy, verbose=verbose, **kwargs)
    return batch_results


def main(args):
    def find_npz_files(data_dir):
        npz_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npz'):
                    npz_files.append(os.path.relpath(os.path.join(root, file), data_dir))
        return npz_files

    torch.manual_seed(42)
    config = FLAGS.config

    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    model = ScoreModelFC(
        config,
        n_poses=N_POSES,
        pose_dim=POSE_DIM,
        hidden_dim=config.model.HIDDEN_DIM,
        embed_dim=config.model.EMBED_DIM,
        n_blocks=config.model.N_BLOCKS,
    )
    model.to(args.device)
    model.eval()
    map_location = {'cuda:0': args.device}
    checkpoint = torch.load(args.ckpt_path, map_location=map_location)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema'])
    state = dict(optimizer=None, model=model, ema=ema, step=0)
    state['step'] = checkpoint['step']
    print(f"=> loaded checkpoint '{args.ckpt_path}' (step {state['step']})")
    ema.copy_to(model.parameters())

    if args.file_path is not None:
        os.makedirs(args.outpath_folder, exist_ok=True)
        batch_results = denoise(config, args, model, args.file_path, args.outpath_folder, std=args.noise_std)
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

        print('Test dataset consists of {} sequences'.format(len(seqs)))

        # Initialize a dictionary to store all batch results
        all_batch_results = {}
        # logging
        csv_file_path = os.path.join(args.outpath_folder, f'{args.dataset}_results_logging.csv')
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Sequence', 'init_MPJPE', 'MPJPE', 'MPVPE'])

        for seq in seqs:
            gt_path = os.path.join(data_dir, seq)
            batch_results = denoise(config, args, model, gt_path, out_path, std=args.noise_std, verbose=True)

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
