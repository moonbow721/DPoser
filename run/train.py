import os
import pprint
import traceback
from pathlib import Path

import cv2
import numpy as np

import torch
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader

from lib.body_model.visual import save_obj, render_mesh
from lib.utils.metric import average_pairwise_distance
from lib.utils.misc import create_mask

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.utils.generic import create_logger

from lib.algorithms.advanced.model import ScoreModelFC, TimeMLPs
from lib.algorithms.advanced import losses, sde_lib, sampling, likelihood
from lib.algorithms.ema import ExponentialMovingAverage

from lib.dataset.AMASS import AMASSDataset, N_POSES
from lib.utils.transforms import rot6d_to_axis_angle
from lib.body_model.body_model import BodyModel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])

# global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='train diffusion model')
    parser.add_argument('--dataset-folder', type=str, default='./data/AMASS/amass_processed',
                        help='the folder includes necessary normalizing parameters')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='path of SMPLX model [for visual validation]')

    parser.add_argument('--restore-dir', type=str, help='resume training')
    parser.add_argument('--shape', type=bool, default=False, help='handle human shapes (have not been tested)')
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--task', type=str, default=None, help='for validating')
    parser.add_argument('--name', type=str, default='', help='name of checkpoint folder')

    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(root_path='', subset='train', version='', sample_interval=None,
                   rot_rep='rot6d', return_shape=False, normalize=True, min_max=True):
    dataset = AMASSDataset(root_path=root_path,
                           version=version, subset=subset, sample_interval=sample_interval,
                           rot_rep=rot_rep, return_shape=return_shape, normalize=normalize, min_max=min_max)
    print('AMASS version: {}, rot_rep: {}, normalize: {}'.format(version, rot_rep, normalize))

    # drop the last batch to ensure that body model can work all the time
    if subset == 'train':
        dataloader = DataLoader(dataset,
                                batch_size=FLAGS.config.training.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=False,
                                drop_last=True)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=FLAGS.config.eval.batch_size,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=False,
                                drop_last=True)

    return dataloader, dataset


def main(args):
    def log_metrics(metrics, step, config, logger):
        log_freq = config.training.log_freq
        msg = f'Iter: [{step}/{num_train_steps}, {step / num_train_steps * 100:.2f}%][{idx}/{len(train_loader)}],\t'
        for key, value in metrics.items():
            metrics[key] /= log_freq
            msg += f"{key}: {metrics[key]:.6f},\t"
        logger.info(msg)
        metrics = {key: 0.0 for key in metrics}
        return metrics

    def log_eval_metrics(metrics, step, writer):
        for key, value in metrics.items():
            avg_value = np.mean(value).item()
            writer.add_scalar(f'eval_{key}', avg_value, step)
            metrics[key] = []  # Reset for the next evaluation

    # args = parse_args()
    config = FLAGS.config

    logger, final_output_dir, tb_log_dir = create_logger(
        config, 'train', folder_name=args.name)
    if config.training.render:
        obj_dir = Path(final_output_dir) / 'obj_results'
        render_dir = Path(final_output_dir) / 'render_results'
        if not obj_dir.exists():
            print('=> creating {}'.format(obj_dir))
            obj_dir.mkdir()
        if not render_dir.exists():
            print('=> creating {}'.format(render_dir))
            render_dir.mkdir()

    logger.info(pprint.pformat(config))
    logger.info(pprint.pformat(args))
    writer = SummaryWriter(tb_log_dir)

    ''' setup body model for val'''
    body_model_vis = BodyModel(bm_path=args.bodymodel_path,
                               num_betas=10,
                               batch_size=50,
                               model_type='smplx').to(device)

    ''' setup datasets, dataloaders'''
    if args.sample:
        logger.info(f'sample trainset every {args.sample} frame')

    train_loader, train_dataset = get_dataloader(args.dataset_folder, 'train', args.version, args.sample, config.data.rot_rep,
                                                 args.shape, config.data.normalize, config.data.min_max)
    test_loader, test_dataset = get_dataloader(args.dataset_folder, 'test', args.version, 100, config.data.rot_rep,
                                               args.shape, config.data.normalize,
                                               config.data.min_max)  # always sample testset to save time
    denormalize_data = train_dataset.Denormalize if config.data.normalize else lambda x: x

    logger.info(f'total train samples: {len(train_dataset)}')
    logger.info(f'total test samples: {len(test_dataset)}')

    ''' setup score networks '''
    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    if config.model.type == 'ScoreModelFC':
        model = ScoreModelFC(
            config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=config.model.HIDDEN_DIM,
            embed_dim=config.model.EMBED_DIM,
            n_blocks=config.model.N_BLOCKS,
        )
    elif config.model.type == 'TimeMLPs':
        model = TimeMLPs(
            config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=config.model.HIDDEN_DIM,
            n_blocks=config.model.N_BLOCKS,
        )
    else:
        raise NotImplementedError('unsupported model')

    model.to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())

    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)  # based on iteration instead of epochs

    # auto resume
    start_epoch = 0
    if args.restore_dir and os.path.exists(args.restore_dir):
        ckpt_path = os.path.join(args.restore_dir, 'checkpoint-step55000.pth')
        logger.info(f'=> loading checkpoint: {ckpt_path}')

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        ema.load_state_dict(checkpoint['ema'])
        state['step'] = checkpoint['step']

        logger.info(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")

    # Identity func
    scaler = lambda x: x
    inverse_scaler = lambda x: x

    # Setup SDEs
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

    sampling_shape = (config.eval.batch_size, N_POSES * POSE_DIM)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # Build one-step training and evaluation functions
    if config.training.auxiliary_loss:
        body_model_train = BodyModel(bm_path=args.bodymodel_path,
                                     num_betas=10,
                                     batch_size=FLAGS.config.training.batch_size,
                                     model_type='smplx').to(device)
        kwargs = {'denormalize': denormalize_data, 'body_model': body_model_train,
                  'rot_rep': config.data.rot_rep, 'denoise_steps': config.training.denoise_steps}
    else:
        kwargs = {}
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=config.training.reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting,
                                       auxiliary_loss=config.training.auxiliary_loss, **kwargs)

    likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler, rtol=1e-4, atol=1e-4, eps=1e-4)

    num_train_steps = config.training.n_iters
    step = state['step']
    epoch = start_epoch
    metrics = {}
    best_APD = 0.0

    try:
        ''' training loop '''
        while step < num_train_steps:
            model.train()
            for idx, batch_data in enumerate(train_loader):
                poses = batch_data['poses'].to(device, non_blocking=True)
                loss_dict = train_step_fn(state, batch=poses, condition=None, mask=None)
                # Update and log metrics
                for key, value in loss_dict.items():
                    if key not in metrics:
                        metrics[key] = 0.0
                    metrics[key] += value.item()
                    writer.add_scalar(key, value.item(), step)
                step = state['step']
                if step % config.training.log_freq == 0:
                    metrics = log_metrics(metrics, step, config, logger)

                '''
                ******************    validating start     ******************
                '''
                if step % config.training.eval_freq == 0:
                    print('start validating')
                    # sampling process
                    model.eval()
                    with torch.no_grad():
                        all_results = []
                        eval_metrics = {'bpd': [], 'mpvpe_all': [], 'mpjpe_body': []}

                        for idx, batch_data in enumerate(test_loader):
                            poses = batch_data['poses'].to(device, non_blocking=True)
                            batch_size = poses.shape[0]
                            # Generate and save samples
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())

                            '''     ******* task1 bpd *******     '''
                            bpd, z, nfe = likelihood_fn(model, poses)
                            logger.info(f'Sample bpd: {bpd.mean().item()} with nfe: {nfe}')
                            eval_metrics['bpd'].append(bpd.mean().item())

                            '''     ******* task2 completion *******     '''
                            mask, observation = create_mask(poses, part='left_leg')

                            hypo_num = 5
                            args.task = 'completion'
                            multihypo_denoise = []
                            for hypo in range(hypo_num):
                                _, completion = sampling_fn(model, observation=observation, mask=mask, args=args)
                                multihypo_denoise.append(completion)
                            multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

                            preds = denormalize_data(multihypo_denoise)
                            gts = denormalize_data(poses)
                            if config.data.rot_rep == 'rot6d':
                                preds = rot6d_to_axis_angle(preds.reshape(-1, 6)).reshape(batch_size, hypo_num, -1)
                                gts = rot6d_to_axis_angle(gts.reshape(-1, 6)).reshape(batch_size, -1)
                            from lib.dataset.AMASS import Evaler
                            evaler = Evaler(body_model=body_model_vis, part='left_leg')
                            eval_results = evaler.multi_eval_bodys(preds, gts)
                            logger.info('Sample mpvpe_all: {}'.format(np.mean(eval_results['mpvpe_all'])))
                            logger.info('Sample mpjpe_body: {}'.format(np.mean(eval_results['mpjpe_body'])))
                            eval_metrics['mpvpe_all'].append(eval_results['mpvpe_all'].mean().item())
                            eval_metrics['mpjpe_body'].append(eval_results['mpjpe_body'].mean().item())

                            '''      ******* task3 generation *******     '''
                            trajs, samples = sampling_fn(
                                model,
                                observation=None
                            )  # [t, b, j*6], [b, j*6]
                            ema.restore(model.parameters())
                            all_results.append(samples)

                        log_eval_metrics(eval_metrics, step, writer)

                    slice_step = sde.N // 10
                    trajs = trajs[::slice_step, :5, ]  # [10time, 5sample, j*6]
                    all_results = torch.cat(all_results, dim=0)[:50]  # [50, j*6]

                    trajs = denormalize_data(trajs)
                    all_results = denormalize_data(all_results)

                    if config.data.rot_rep == 'rot6d':
                        trajs = rot6d_to_axis_angle(trajs.reshape(-1, 6)).reshape(-1,
                                                                                  N_POSES * 3)  # -> [10time*5sample, j*3]
                        all_results = rot6d_to_axis_angle(all_results.reshape(-1, 6)).reshape(-1,
                                                                                              N_POSES * 3)  # -> [50, j*3]

                    '''      ******* compute APD *******      '''
                    body_out = body_model_vis(pose_body=all_results)
                    joints3d = body_out.Jtr
                    body_joints3d = joints3d[:, :22, :]
                    APD = average_pairwise_distance(body_joints3d)
                    logger.info(f'APD: {APD.item()}')
                    writer.add_scalar('APD', APD.item(), step)

                    if config.training.render:
                        # saving trajs
                        body_out = body_model_vis(pose_body=trajs)
                        meshes = body_out.v.detach().cpu().numpy().reshape(10, 5, -1, 3)  # [50, N, 3] -> [10, 5, N, 3]
                        faces = body_out.f.cpu().numpy()  # [F, 3]
                        for sample_idx in range(5):
                            for time_idx in range(10):
                                mesh = meshes[time_idx, sample_idx]
                                save_obj(mesh, faces, os.path.join(obj_dir, 'sample{}_time{}.obj'.format(sample_idx + 1,
                                                                                                         time_idx + 1)))
                                rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt})
                                cv2.imwrite(os.path.join(render_dir, 'render_sample{}_time{}.jpg'.format(sample_idx + 1,
                                                                                                         time_idx + 1)),
                                            rendered_img)
                        # saving samples
                        body_out = body_model_vis(pose_body=all_results)
                        meshes = body_out.v.detach().cpu().numpy()  # [50, N, 3]
                        faces = body_out.f.cpu().numpy()  # [F, 3]

                        for sample_idx in range(50):
                            mesh = meshes[sample_idx]
                            save_obj(mesh, faces, os.path.join(obj_dir, 'Rsample{}.obj'.format(sample_idx + 1)))
                            rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt})
                            cv2.imwrite(os.path.join(render_dir, 'Rrender_sample{}.jpg'.format(sample_idx + 1)),
                                        rendered_img)

                        del body_out, meshes, mesh, faces
                    save_path = os.path.join(final_output_dir, 'last_samples.npz')
                    logger.info(f'save eval samples to {save_path}')
                    np.savez(save_path,
                             **{'pose_trajs': trajs.cpu().numpy().reshape(10, 5, -1),
                                'pose_samples': all_results.cpu().numpy().reshape(1, 50, -1), }
                             )

                    print('validating completed')

                    '''
                    ******************    validating end      ******************
                    '''

                    if APD.item() > best_APD:
                        # update best checkpoint
                        best_APD = APD.item()
                        logger.info('saving best checkpoint, APD: {}'.format(best_APD))
                        torch.save(
                            {
                                'model_state_dict': model.state_dict(),
                                'epoch': epoch + 1,
                                'ema': state['ema'].state_dict(),
                                'step': state['step'],
                            },
                            os.path.join(final_output_dir, 'best_model.pth')
                        )

                # log and save ckpt
                if step % config.training.save_freq == 0:
                    logger.info(f'Save checkpoint to {final_output_dir}')
                    save_dict = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ema': state['ema'].state_dict(),
                        'step': state['step'],
                    }
                    torch.save(save_dict,
                               os.path.join(final_output_dir, 'checkpoint-step{}.pth').format(state['step']))
                epoch += 1

    except Exception as e:
        traceback.print_exc()
    finally:
        writer.close()
        logger.info(f'End. Final output dir: {final_output_dir}')


if __name__ == '__main__':
    torch.manual_seed(42)
    app.run(main, flags_parser=parse_args)
