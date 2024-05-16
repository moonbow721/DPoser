import os
from functools import partial
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from absl import flags, app
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags

from lib.algorithms.advanced import likelihood, sde_lib, sampling
from lib.algorithms.advanced.model import create_model
from lib.algorithms.completion import DPoserComp
from lib.body_model.body_model import BodyModel
from lib.body_model.visual import render_mesh, multiple_render
from lib.dataset.body import N_POSES, Evaler
from lib.utils.generic import load_pl_weights, load_model
from lib.utils.metric import evaluate_fid, evaluate_prdc, average_pairwise_distance, self_intersections_percentage
from lib.utils.misc import create_mask

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Visualizing configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

from lib.dataset.utils import Posenormalizer

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='visualize the save files and demo on toy data')

    parser.add_argument('--file-path', type=str, default='./examples/toy_body_data.npz', help='saved npz file')
    parser.add_argument('--task', type=str, default='view', choices=['view',
                                                                     'generation',
                                                                     'eval_generation',
                                                                     'generation_process',
                                                                     'completion', ])
    parser.add_argument('--hypo', type=int, default=10, help='multi hypothesis prediction for completion')
    parser.add_argument('--part', type=str, default='left_leg', choices=['left_leg', 'right_leg', 'left_arm',
                                                                         'right_arm', 'trunk', 'hands', 'legs', 'arms'],
                        help='the masked part for completion task')
    parser.add_argument('--mode', default='DPoser', choices=['DPoser', 'ScoreSDE', 'MCG', 'DPS'],
                        help='different solvers for completion task')

    parser.add_argument('--dataset-folder', type=str,
                        default='../data/human/Bodydataset/amass_processed', help='dataset root')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX for visualization')
    parser.add_argument('--ckpt-path', type=str,
                        default='./pretrained_models/amass/BaseMLP/epoch=36-step=150000-val_mpjpe=38.17.ckpt',
                        help='load trained diffusion model')

    parser.add_argument('--view', type=str, default='front', help='render direction')
    parser.add_argument('--faster', action='store_true', help='faster render (lower quality)')
    parser.add_argument('--output-path', type=str, default='./output/body/test_results')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args(argv[1:])

    return args


def main(args):
    torch.manual_seed(42)
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
    model = create_model(config.model, N_POSES, POSE_DIM)
    model.to(args.device)
    model.eval()
    load_model(model, config, args.ckpt_path, args.device, is_ema=True)

    inverse_scaler = lambda x: x
    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, rtol=1e-4, atol=1e-4, eps=1e-4)

    Normalizer = Posenormalizer(data_path=f'{args.dataset_folder}/{args.version}/train',
                                normalize=config.data.normalize,
                                min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)
    denormalize_fn = Normalizer.offline_denormalize

    if args.task == 'generation':
        target_path = os.path.join(args.output_path, 'generation')

        sample_num = 100
        sampling_shape = (sample_num, N_POSES * POSE_DIM)
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=sample_num,
                               model_type='smplx').to(args.device)

        save_renders(samples, denormalize_fn, body_model, target_path, 'generated_sample{}.png', faster=args.faster)
        print(f'samples saved under {target_path}')
        return

    elif args.task == 'eval_generation':
        sample_num = 50000
        sampling_shape = (sample_num, N_POSES * POSE_DIM)
        sampling_eps = 5e-3
        sampling.method = 'pc'  # pc or ode
        # config.sampling.corrector = 'langevin'
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                                   device=args.device)
        _, samples = default_sampler(model, observation=None)
        samples = denormalize_fn(samples, to_axis=True)
        fid = evaluate_fid(samples, f'{args.dataset_folder}/{args.version}/statistics.npz')
        print('FID for 50000 generated samples', fid)
        prdc = evaluate_prdc(samples, f'{args.dataset_folder}/{args.version}/reference_batch.npz')
        print(prdc)

        samples = samples[:500]
        body_model = BodyModel(bm_path='/data3/ljz24/projects/3d/body_models/smpl',
                               model_type='smpl',
                               batch_size=500,
                               num_betas=10).to(args.device)

        zero_hands = torch.zeros([500, 6]).to(args.device)  # the gap between smpl and smplx body
        full_samples = torch.cat([samples, zero_hands], dim=1)
        body_out = body_model(body_pose=full_samples)
        joints3d = body_out.Jtr
        body_joints3d = joints3d[:, :22, :]
        APD = average_pairwise_distance(body_joints3d)
        SI = self_intersections_percentage(body_out.v, body_out.f).mean().item()
        print('average_pairwise_distance for 500 generated samples', APD)
        print('self-intersections percentage for 500 generated samples', SI)

        return

    elif args.task == 'generation_process':
        target_path = os.path.join(args.output_path, 'generation_process')

        video_num = 10
        sampling_shape = (video_num, N_POSES * POSE_DIM)

        # config.sampling.probability_flow = True
        assert config.sampling.method == 'pc'  # we don't save trajectories for ode sampler
        default_sampler = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler,
                                                   sampling_eps, device=args.device)
        trajs, _ = default_sampler(model, observation=None, gather_traj=True)
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=100,
                               model_type='smplx').to(args.device)

        for idx in range(video_num):
            traj = trajs[9::10, idx]  # [100, j*6]
            num_frame = traj.shape[0]
            traj = denormalize_fn(traj, to_axis=True)
            body_out = body_model(body_pose=traj)
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
    sample_num = 20
    body_poses = data['pose_samples'][:sample_num]
    print(f'loaded axis pose data {body_poses.shape} from {args.file_path}')
    body_poses = torch.from_numpy(body_poses).to(args.device)

    if args.task == 'view':
        target_path = os.path.join(args.output_path, 'view')
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        save_renders(body_poses, None, body_model, target_path, 'GT_sample{}.png', convert=False, faster=args.faster)
        print(f'rendered images saved under {target_path}')
        return

    elif args.task == 'completion':
        task_args = SimpleNamespace(task=None)
        if args.mode == 'DPS':
            task_args.task, inverse_solver = 'default', 'BP'
        elif args.mode == 'MCG':
            task_args.task, inverse_solver = 'completion', 'BP'
        elif args.mode == 'ScoreSDE':
            task_args.task, inverse_solver = 'completion', None
        else:  # plain generation sampler
            task_args.task, inverse_solver = 'default', None
        comp_sampler = sampling.get_sampling_fn(config, sde, body_poses.shape, inverse_scaler, sampling_eps,
                                                device=args.device, inverse_solver=inverse_solver)

        comp_fn = DPoserComp(model, sde, config.training.continuous, batch_size=sample_num, improve_baseline=True)
        body_model = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=sample_num,
                               model_type='smplx').to(args.device)

        gts = body_poses
        body_poses = Normalizer.offline_normalize(body_poses, from_axis=True)  # [b, N_POSES*6]
        mask, observation = create_mask(body_poses, part=args.part, model='body', observation_type='noise')

        hypo_num = args.hypo
        multihypo_denoise = []
        for hypo in range(hypo_num):
            if args.mode == 'DPoser':
                completion = comp_fn.optimize(observation, mask, lr=0.1)
            else:
                _, completion = comp_sampler(model, observation=observation, mask=mask, args=task_args)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = denormalize_fn(multihypo_denoise, to_axis=True)

        evaler = Evaler(body_model=body_model, part=args.part)
        eval_results = evaler.multi_eval_bodys_all(preds, gts)
        evaler.print_eval_result_all(eval_results, )

        target_path = os.path.join(args.output_path, 'completion', args.part)
        save_renders(gts, denormalize_fn, body_model, target_path, 'sample{}_original.png', convert=False,
                     faster=args.faster, view=args.view)
        print(f'Original samples under {target_path}')
        save_renders(observation, denormalize_fn, body_model, target_path, 'sample{}_masked.png', faster=args.faster,
                     view=args.view)
        print(f'Masked samples under {target_path}')
        for hypo in range(hypo_num):
            save_renders(multihypo_denoise[:, hypo], denormalize_fn, body_model, target_path,
                         'sample{}_completion' + str(hypo) + '.png', faster=args.faster, view=args.view)
        print(f'Completion samples under {target_path}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    app.run(main, flags_parser=parse_args)