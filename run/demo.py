import os
from functools import partial

import cv2
import math
import numpy as np
import torch
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from torch import nn

from lib.algorithms.advanced import likelihood, sde_lib, sampling
from lib.algorithms.advanced.model import ScoreModelFC
from lib.algorithms.ema import ExponentialMovingAverage
from lib.body_model.body_model import BodyModel
from lib.body_model.visual import render_mesh, multiple_render
from lib.dataset.AMASS import N_POSES
from lib.utils.metric import average_pairwise_distance, self_intersections_percentage
from lib.utils.misc import create_mask, linear_interpolation, slerp_interpolation

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

from lib.dataset.AMASS import Posenormalizer

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='visualize the save files and demo on toy data')

    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/axis-zscore-400k.pth')
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='path of SMPLX model')
    parser.add_argument('--dataset-folder', type=str, default='./data/AMASS/amass_processed',
                        help='the folder includes necessary normalizing parameters')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')

    parser.add_argument('--file-path', type=str, default='./examples/toy_data.npz', help='saved npz file')
    parser.add_argument('--task', type=str, default='view',
                        choices=['view',
                                 'generation',
                                 'generation_process',
                                 'interpolation',
                                 'completion',  # generation with replacement (ScoreSDE, MCG, DPS)
                                 'completion2',  # optimization (DPoser)
                                 ])
    parser.add_argument('--metrics', action='store_true', help='compute APD and SI for generation tasks')
    parser.add_argument('--hypo', type=int, default=10, help='multi hypothesis prediction for completion')
    parser.add_argument('--part', type=str, default='left_leg', choices=['left_leg', 'right_leg', 'left_arm',
                                                                         'right_arm', 'trunk', 'hands',
                                                                         'legs', 'arms'],
                        help='the masked part for completion task')
    parser.add_argument('--view', type=str, default='front', help='render direction')
    parser.add_argument('--faster', action='store_true', help='faster render (lower quality)')
    parser.add_argument('--video', action='store_true', help='save videos for interpolation')
    parser.add_argument('--output-path', type=str, default='./output/test_results')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args(argv[1:])

    return args


def main(args):
    """
    *****************        load some gt samples and view       *****************
    """
    save_renders = partial(multiple_render, bg_img=bg_img, focal=focal, princpt=princpt, device=args.device)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    sample_num = 50
    body_model = BodyModel(bm_path=args.bodymodel_path,
                           num_betas=10,
                           batch_size=sample_num,
                           model_type='smplx').to(args.device)

    """
    *****************        model preparation for demo tasks       *****************    
    """
    config = FLAGS.config
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                               N=config.model.num_scales)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                            N=config.model.num_scales)
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")
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
    ema.copy_to(model.parameters())

    inverse_scaler = lambda x: x
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, rtol=1e-4, atol=1e-4, eps=1e-4)
    # choose some anchor poses, then we interpolate immediate poses between the adjacent poses
    Normalizer = Posenormalizer(data_path=f'{args.dataset_folder}/{args.version}/train',
                                normalize=config.data.normalize,
                                min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)

    if args.task == 'generation':
        target_path = os.path.join(args.output_path, 'generation')
        sampling_shape = (sample_num, N_POSES * POSE_DIM)
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        save_renders(samples, Normalizer, body_model, target_path, 'generated_sample{}.png', faster=args.faster)
        print(f'samples saved under {target_path}')

        # evaluate APD and SI as metrics
        if args.metrics:
            sample_num = 500
            sampling_shape = (sample_num, N_POSES * POSE_DIM)
            sampling_eps = 5e-3
            sampling.method = 'pc'  # pc or ode
            config.sampling.corrector = 'langevin'
            default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                       device=args.device)
            _, samples = default_sampler(model, observation=None)

            # Our paper use SMPL model to evaluate APD and SI, following Pose-NDF
            body_model = BodyModel(bm_path='../body_models/smpl',
                                   model_type='smpl',
                                   batch_size=sample_num,
                                   num_betas=10).to(args.device)
            samples = Normalizer.offline_denormalize(samples, to_axis=True)
            zero_hands = torch.zeros([sample_num, 6]).to(args.device)  # the gap between smpl and smplx body
            samples = torch.cat([samples, zero_hands], dim=1)
            body_out = body_model(pose_body=samples)
            joints3d = body_out.Jtr
            body_joints3d = joints3d[:, :22, :]
            APD = average_pairwise_distance(body_joints3d)
            SI = self_intersections_percentage(body_out.v, body_out.f).mean().item()
            print('average_pairwise_distance for 500 generated samples', APD)
            print('self-intersections percentage for 500 generated samples', SI)
        return

    elif args.task == 'generation_process':
        target_path = os.path.join(args.output_path, 'generation_process')
        os.makedirs(target_path, exist_ok=True)

        video_num = 3
        sampling_shape = (video_num, N_POSES * POSE_DIM)

        # config.sampling.probability_flow = True
        # config.sampling.corrector = 'langevin'
        assert config.sampling.method == 'pc'  # we don't save trajectories for ode sampler
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        trajs, _ = default_sampler(model, observation=None)
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=100,
                               model_type='smplx').to(args.device)

        for idx in range(video_num):
            traj = trajs[9::10, idx]  # [100, j*6]
            num_frame = traj.shape[0]
            traj = Normalizer.offline_denormalize(traj, to_axis=True)
            body_out = body_model(pose_body=traj)
            meshes = body_out.v.detach().cpu().numpy()
            faces = body_out.f.cpu().numpy()
            all_frames = []
            for frame in range(num_frame):
                mesh = meshes[frame]
                rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt}, view='front')
                all_frames.append(rendered_img)

            # save the results to a video
            height, width, layers = all_frames[0].shape
            fps = 30
            video_path = os.path.join(target_path, "generation_process{}.mp4".format(idx))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for i in range(len(all_frames)):
                frame = (all_frames[i]).astype(np.uint8)
                out.write(frame)

            out.release()
            print(f"Video saved at {video_path}")

        return

    """
    *****************        data preparation for demo tasks       *****************    
    """
    data = np.load(args.file_path, allow_pickle=True)
    body_poses = data['pose_samples'][:sample_num]
    print(f'loaded axis pose data {body_poses.shape} from {args.file_path}')
    body_poses = torch.from_numpy(body_poses).to(args.device)
    body_model = BodyModel(bm_path=args.bodymodel_path,
                           num_betas=10,
                           batch_size=sample_num,
                           model_type='smplx').to(args.device)

    if args.task == 'view':
        target_path = os.path.join(args.output_path, 'view')
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        save_renders(body_poses, None, body_model, target_path, 'GT_sample{}.png', convert=False, faster=args.faster)
        print(f'rendered images saved under {target_path}')
        return

    elif args.task == 'completion':
        torch.set_grad_enabled(True)
        from lib.algorithms.advanced import utils as mutils
        class DPoserComp(object):
            def __init__(self, diffusion_model, sde, continuous, batch_size=1):
                self.batch_size = batch_size
                self.sde = sde
                self.score_fn = mutils.get_score_fn(sde, diffusion_model, train=False, continuous=continuous)
                self.rsde = sde.reverse(self.score_fn, False)
                # L2 loss
                self.loss_fn = nn.MSELoss(reduction='none')
                self.data_loss = nn.MSELoss(reduction='mean')

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
                    x_current = alpha_before / alpha_current * (
                            x_current - sigma_current[:, None] * score) + sigma_before[
                                                                          :,
                                                                          None] * score
                alpha, sigma = self.sde.return_alpha_sigma(time_traj[0])
                SNR = alpha / sigma[:, None]
                return x_current.detach(), SNR

            def loss(self, x_0, vec_t, weighted=False, multi_denoise=False):
                # x_0: [B, j*6], vec_t: [B],
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

                dposer_loss = torch.mean(weight * self.loss_fn(x_0, denoise_data))

                return dposer_loss

            def get_loss_weights(self):
                """Set loss weights"""
                loss_weight = {'data': lambda cst, it: 100 * cst / (1 + it),
                               'dposer': lambda cst, it: 0.1 * cst * (it + 1)}
                return loss_weight

            @staticmethod
            def backward_step(loss_dict, weight_dict, it):
                w_loss = dict()
                for k in loss_dict:
                    w_loss[k] = weight_dict[k](loss_dict[k], it)

                tot_loss = list(w_loss.values())
                tot_loss = torch.stack(tot_loss).sum()
                return tot_loss

            def optimize(self, observation, mask, time_strategy='2', lr=0.1,
                         sample_trun=5.0, sample_time=900, iterations=2, steps_per_iter=100):
                total_steps = iterations * steps_per_iter
                opti_variable = observation.clone().detach()
                opti_variable.requires_grad = True
                optimizer = torch.optim.Adam([opti_variable], lr, betas=(0.9, 0.999))
                weight_dict = self.get_loss_weights()
                loss_dict = dict()

                eps = 1e-3
                timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=observation.device)
                for it in range(iterations):
                    for i in range(steps_per_iter):
                        step = it * steps_per_iter + i
                        optimizer.zero_grad()

                        '''   *************      DPoser loss ***********         '''
                        if time_strategy == '1':  # not recommend
                            quan_t = torch.randint(self.sde.N, [1])
                        elif time_strategy == '2':
                            quan_t = torch.tensor(sample_time)
                        elif time_strategy == '3':
                            quan_t = self.sde.N - math.floor(
                                torch.tensor(total_steps - step - 1) * (self.sde.N / (sample_trun * total_steps))) - 2
                        else:
                            raise NotImplementedError('unsupported time sampling strategy')

                        t = timesteps[quan_t]
                        vec_t = torch.ones(self.batch_size, device=observation.device) * t
                        loss_dict['dposer'] = self.loss(opti_variable, vec_t, quan_t)
                        loss_dict['data'] = self.data_loss(opti_variable * mask, observation * mask)
                        '''   ***********      DPoser loss   ************       '''

                        # Get total loss for backward pass
                        tot_loss = self.backward_step(loss_dict, weight_dict, it)
                        tot_loss.backward()
                        optimizer.step()

                return opti_variable

        compfn = DPoserComp(model, sde, config.training.continuous, batch_size=sample_num)

        target_path = os.path.join(args.output_path, 'completion')
        save_renders(body_poses, Normalizer, body_model, target_path, 'sample{}_original.jpg', convert=False,
                     faster=args.faster, view=args.view)
        print(f'Original samples under {target_path}')

        gts = body_poses
        body_poses = Normalizer.offline_normalize(body_poses, from_axis=True)  # [b, N_POSES*6]
        mask, observation = create_mask(body_poses, part=args.part)

        hypo_num = args.hypo
        multihypo_denoise = []
        for hypo in range(hypo_num):
            completion = compfn.optimize(observation, mask)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = Normalizer.offline_denormalize(multihypo_denoise, to_axis=True)
        from lib.dataset.AMASS import Evaler
        evaler = Evaler(body_model=body_model, part=args.part)
        eval_results = evaler.multi_eval_bodys(preds, gts)
        evaler.print_multi_eval_result(eval_results, hypo_num)

        save_renders(observation, Normalizer, body_model, target_path, 'sample{}_masked.png', faster=args.faster,
                     view=args.view)
        print(f'Masked samples under {target_path}')
        for hypo in range(hypo_num):
            save_renders(multihypo_denoise[:, hypo], Normalizer, body_model, target_path,
                         'sample{}_completion' + str(hypo) + '.png', faster=args.faster, view=args.view)
        print(f'Completion samples under {target_path}')

    elif args.task == 'completion2':
        # (generation + subspace projection) as default solver, ScoreSDE
        target_path = os.path.join(args.output_path, 'completion')
        save_renders(body_poses, Normalizer, body_model, target_path, 'sample{}_original.png',
                     convert=False, faster=args.faster, view=args.view)
        print(f'Original samples under {target_path}')

        gts = body_poses
        body_poses = Normalizer.offline_normalize(body_poses, from_axis=True)  # [b, N_POSES*6]
        mask, observation = create_mask(body_poses, part=args.part)

        comp_sampler = sampling.get_sampling_fn(config, sde, observation.shape, inverse_scaler, sampling_eps,
                                                device=args.device)
        hypo_num = args.hypo
        multihypo_denoise = []
        for hypo in range(hypo_num):
            _, completion = comp_sampler(model, observation=observation, mask=mask, args=args)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = Normalizer.offline_denormalize(multihypo_denoise, to_axis=True)
        from lib.dataset.AMASS import Evaler
        evaler = Evaler(body_model=body_model, part=args.part)
        eval_results = evaler.multi_eval_bodys(preds, gts)
        evaler.print_multi_eval_result(eval_results, hypo_num)

        save_renders(observation, Normalizer, body_model, target_path, 'sample{}_masked.png',
                     faster=args.faster, view=args.view)
        print(f'Masked samples under {target_path}')
        for hypo in range(hypo_num):
            save_renders(multihypo_denoise[:, hypo], Normalizer, body_model, target_path,
                         'sample{}_completion' + str(hypo) + '.png', faster=args.faster, view=args.view)
        print(f'Completion samples under {target_path}')

    elif args.task == 'interpolation':
        target_path = os.path.join(args.output_path, 'interpolation')

        inter_frames = 60
        choosen_idx = [1, 10, 11, 12, 17, 14]

        anchor_poses = body_poses[choosen_idx]
        anchor_num = anchor_poses.shape[0]
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=anchor_num,
                               model_type='smplx').to(args.device)

        save_renders(anchor_poses, Normalizer, body_model, target_path, 'original_sample{}.png', convert=False,
                     idx_map=choosen_idx, faster=args.faster)
        print(f'Original samples under {target_path}')

        assert len(anchor_poses.shape) == 2
        anchor_poses = Normalizer.offline_normalize(anchor_poses, from_axis=True)
        # encode poses to the ODE latents of diffusion models
        _, anchor_z, __ = likelihood_fn(model, anchor_poses)

        '''
        *****************        flow back along the learned ODE       *****************     
        '''
        # ODE trajectories to generate from deterministic latents
        # TODO: try different samplers here
        config.sampling.probability_flow = True  # necessary
        config.sampling.method = 'pc'
        config.sampling.predictor = 'euler_maruyama'
        config.sampling.corrector = 'none'  # Note: corrector should never be used for deterministic sampling
        eps = 1e-5

        sampling_shape = (anchor_num, N_POSES * POSE_DIM)
        deterministic_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eps,
                                                         device=args.device)
        _, samples = deterministic_sampler(
            model,
            z=anchor_z
        )
        save_renders(samples, Normalizer, body_model, target_path, 'reconstruction_sample{}.png',
                     idx_map=choosen_idx, faster=args.faster)
        print(f'Reconstruction samples under {target_path}')

        '''
        *****************        interpolate and save videos       *****************
        '''
        sampling_shape = (inter_frames, N_POSES * POSE_DIM)
        deterministic_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eps,
                                                         device=args.device)
        all_renders = []
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=inter_frames,
                               model_type='smplx').to(args.device)
        for idx in range(anchor_num - 1):
            inter_latents = slerp_interpolation(anchor_z[idx], anchor_z[idx + 1], inter_frames)
            _, samples = deterministic_sampler(
                model,
                z=inter_latents
            )

            if args.video:
                samples = Normalizer.offline_denormalize(samples, to_axis=True)
                body_out = body_model(pose_body=samples)
                meshes = body_out.v.detach().cpu().numpy()
                faces = body_out.f.cpu().numpy()
                for frame_idx in range(inter_frames):
                    mesh = meshes[frame_idx]
                    rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt}, view='front')
                    all_renders.append(rendered_img)
            else:
                save_renders(samples, Normalizer, body_model, target_path, 'inter_' + str(idx) + '_{}.png',
                             faster=args.faster)

        if args.video:
            # save the results to a video
            height, width, layers = all_renders[0].shape
            fps = 60
            video_path = os.path.join(target_path, "interpolation_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for i in range(len(all_renders)):
                frame = (all_renders[i]).astype(np.uint8)
                out.write(frame)

            out.release()
            print(f"Video saved at {video_path}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    app.run(main, flags_parser=parse_args)