import os
import sys
import argparse

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torchvision
from torch import optim

from lib.body_model.body_model import BodyModel
from lib.body_model.visual import render_mesh
from lib.utils.callbacks import TimerCallback, ModelSizeCallback
from lib.utils.misc import create_mask
from lib.utils.metric import average_pairwise_distance, self_intersections_percentage
from lib.utils.generic import import_configs

from lib.dataset.body import AMASSDataModule, N_POSES, Evaler
from lib.dataset.utils import Posenormalizer

from lib.algorithms.advanced.model import create_model
from lib.algorithms.advanced import losses, sde_lib, sampling, likelihood
from lib.algorithms.ema import ExponentialMovingAverage
from lib.algorithms.completion import DPoserComp
from lib.utils.schedulers import CosineWarmupScheduler


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train diffusion model')
    parser.add_argument('--config-path', '-c', type=str,
                        default='configs.subvp.amass_scorefc_continuous.get_config',
                        help='config files to build DPoser')
    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX for visualization')
    parser.add_argument('--resume-ckpt', '-r', type=str, help='resume training')
    parser.add_argument('--data-root', type=str,
                        default='./body_data', help='dataset root')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--name', type=str, default='default', help='name of checkpoint folder')

    args = parser.parse_args(argv[1:])

    return args


class DPoserTrainer(pl.LightningModule):
    def __init__(self, config,
                 bodymodel_path='',
                 data_path='',
                 N_POSES=21,
                 train_loader=None,
                 val_loader=None, ):
        super().__init__()
        self.config = config
        self.bodymodel_path = bodymodel_path
        self.data_path = data_path
        self.N_POSES = N_POSES
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_hyperparameters(ignore=['train_loader', 'val_loader'])

        # Collect data
        self.last_trajs = None
        self.all_samples = []

        # Initialize the model
        self.POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
        self.model = create_model(config.model, N_POSES, self.POSE_DIM)
        self.model_ema = None
        self.body_model_vis = BodyModel(bm_path=self.bodymodel_path,
                                        num_betas=10,
                                        batch_size=50,
                                        model_type='smplx')
        self.body_model_eval = BodyModel(bm_path=self.bodymodel_path,
                                         num_betas=10,
                                         batch_size=250,
                                         model_type='smplx')
        for param in self.body_model_vis.parameters():
            param.requires_grad = False
        for param in self.body_model_eval.parameters():
            param.requires_grad = False
        self.normalize_fn = None
        self.denormalize_fn = None

        # Setup SDEs and functions
        self.sde = self.setup_sde(config)
        self.sampling_shape = (config.eval.batch_size, N_POSES * self.POSE_DIM)
        self.sampling_eps = 1e-3
        self.train_step_fn = None
        self.sampling_fn = None
        self.likelihood_fn = likelihood.get_likelihood_fn(self.sde, lambda x: x, rtol=1e-4, atol=1e-4, eps=1e-4)
        self.compfn = None

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def setup(self, stage=None):
        if stage == 'fit':
            self.model_ema = ExponentialMovingAverage(self.model.parameters(),
                                                      decay=config.model.ema_rate,
                                                      device=self.device)
            Normalizer = Posenormalizer(
                data_path=self.data_path,
                normalize=self.config.data.normalize,
                min_max=self.config.data.min_max,
                rot_rep=self.config.data.rot_rep,
                device=self.device
            )
            self.normalize_fn = Normalizer.offline_normalize
            self.denormalize_fn = Normalizer.offline_denormalize
            self.train_step_fn = self.setup_step_fn(config)
            self.sampling_fn = sampling.get_sampling_fn(config, self.sde, self.sampling_shape,
                                                        lambda x: x, self.sampling_eps, self.device)

    def setup_step_fn(self, config):
        # Build one-step training and evaluation functions
        kwargs = {}
        if config.training.auxiliary_loss:
            body_model_train = BodyModel(bm_path=self.bodymodel_path,
                                         num_betas=10,
                                         batch_size=config.training.batch_size,
                                         model_type='smplx').to(self.device)
            for param in body_model_train.parameters():
                param.requires_grad = False
            aux_params = {'denormalize': self.denormalize_fn, 'body_model': body_model_train,
                          'model_type': "body", 'denoise_steps': config.training.denoise_steps}
            kwargs.update(aux_params)
        if config.training.random_mask:
            mask_params = {'min_mask_rate': config.training.min_mask_rate,
                           'max_mask_rate': config.training.max_mask_rate,
                           'observation_type': config.training.observation_type}
            kwargs.update(mask_params)

        optimize_fn = losses.optimization_manager(config)
        continuous = config.training.continuous
        likelihood_weighting = config.training.likelihood_weighting
        return losses.get_step_fn(self.sde, train=True, optimize_fn=optimize_fn,
                                  reduce_mean=config.training.reduce_mean, continuous=continuous,
                                  likelihood_weighting=likelihood_weighting,
                                  auxiliary_loss=config.training.auxiliary_loss,  # auxiliary loss
                                  random_mask=config.training.random_mask,  # ambient Diffusion, not used
                                  **kwargs)

    def setup_sde(self, config):
        # Setup SDEs as per your configuration
        if config.training.sde.lower() == 'vpsde':
            return sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                 N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            return sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                    N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            return sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                 N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    def training_step(self, batch, batch_idx):
        poses = self.normalize_fn(batch['body_pose'], from_axis=True)
        # Forward pass and calculate loss
        loss_dict = self.train_step_fn(self.model, batch=poses, condition=None, mask=None)

        # Log the losses
        for key, value in loss_dict.items():
            self.log(f"{key}", value, prog_bar=True, logger=True)

        return loss_dict['loss']  # Assuming 'loss' is a key in your loss_dict

    def on_train_batch_end(self, *args):
        self.model_ema.update(self.model.parameters())

    def on_validation_epoch_start(self) -> None:
        # Store and copy EMA parameters for validation
        self.model_ema.store(self.model.parameters())
        self.model_ema.copy_to(self.model.parameters())
        self.compfn = DPoserComp(self.model, self.sde,
                                 self.config.training.continuous, batch_size=self.config.eval.batch_size)

    def validation_step(self, batch, batch_idx):
        poses = self.normalize_fn(batch['body_pose'], from_axis=True)
        # Process the batch and calculate metrics
        eval_metrics, trajs, samples = self.process_validation_batch(poses)
        self.all_samples.append(samples)

        # Store trajs of the last batch
        if batch_idx == len(self.val_dataloader()) - 1:
            self.last_trajs = trajs

        # Log calculated metrics
        for metric_name, metric_value in eval_metrics.items():
            self.log(f'val_{metric_name}', metric_value, sync_dist=True, logger=True)

        return eval_metrics

    def on_validation_epoch_end(self) -> None:
        self.model_ema.restore(self.model.parameters())
        '''     ******* Compute APD and SI *******     '''
        all_results = torch.cat(self.all_samples, dim=0)[:250]
        body_pose = self.denormalize_fn(all_results, to_axis=True)
        body_out = self.body_model_eval(body_pose=body_pose)
        joints3d = body_out.Jtr
        body_joints3d = joints3d[:, :22, :]
        APD = average_pairwise_distance(body_joints3d)
        SI = self_intersections_percentage(body_out.v, body_out.f).mean()
        self.log('APD', APD.item(), sync_dist=True, logger=True)
        self.log('SI', SI.item(), sync_dist=True, logger=True)

        if self.config.training.render:
            # Use the stored trajs and all_results of the last batch
            self.render_and_log_images(self.last_trajs, all_results)

        # Reset the list of samples
        self.all_samples = []

    @torch.no_grad()
    def process_validation_batch(self, poses):
        eval_metrics = {'bpd': [], 'mpvpe': [], 'mpjpe': []}

        '''     ******* task1 bpd *******     '''
        bpd, z, nfe = self.likelihood_fn(self.model, poses, condition=None)
        eval_metrics['bpd'] = bpd.mean().item()

        '''     ******* task2 completion *******     '''
        mask, observation = create_mask(poses, part='left_leg', model='body')

        hypo_num = 10
        multihypo_denoise = []
        for hypo in range(hypo_num):
            completion = self.compfn.optimize(observation, mask)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = self.denormalize_fn(multihypo_denoise, to_axis=True)
        gts = self.denormalize_fn(poses, to_axis=True)
        evaler = Evaler(body_model=self.body_model_vis, part='left_leg')
        eval_results = evaler.multi_eval_bodys(preds, gts)
        eval_metrics['mpvpe'] = eval_results['mpvpe'].mean().item()
        eval_metrics['mpjpe'] = eval_results['mpjpe'].mean().item()

        '''      ******* task3 generation *******     '''
        trajs, samples = self.sampling_fn(
            self.model,
            observation=None
        )  # [t, b, j*6], [b, j*6]

        return eval_metrics, trajs, samples

    def render_and_log_images(self, trajs, all_results):
        bg_img = np.ones([512, 384, 3]) * 255  # background canvas
        focal = [1500, 1500]
        princpt = [200, 192]

        # Sample some frames for visualization
        slice_step = self.sde.N // 10
        trajs = self.denormalize_fn(trajs[::slice_step, :5, ], to_axis=True).reshape(50, -1)  # [10time, 5sample, j*6]
        all_results = self.denormalize_fn(all_results[:50], to_axis=True)  # [50, j*6]

        # Process and log trajs
        self.process_and_log_meshes(trajs, bg_img, focal, princpt, 'trajs')

        # Process and log samples
        self.process_and_log_meshes(all_results, bg_img, focal, princpt, 'samples')

    def process_and_log_meshes(self, poses, bg_img, focal, princpt, tag_prefix):
        body_out = self.body_model_vis(body_pose=poses)
        meshes = body_out.v.detach().cpu().numpy()
        faces = body_out.f.cpu().numpy()

        rendered_images = []
        for mesh in meshes:
            rendered_img = render_mesh(bg_img, mesh, faces, {'focal': focal, 'princpt': princpt})
            rendered_img_tensor = self.convert_to_tensor(rendered_img)
            rendered_images.append(rendered_img_tensor)

        # Create an image grid and log it
        image_grid = torchvision.utils.make_grid(rendered_images, nrow=10)  # 10 columns
        self.logger.experiment.add_image(f'{tag_prefix}_grid', image_grid, self.current_epoch)

    def convert_to_tensor(self, img):
        # Convert the image to a PyTorch tensor and normalize it to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        return img_tensor

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = self.get_optimizer(self.config, self.model.parameters())

        # Set up the learning rate scheduler
        if self.config.optim.warmup > 0:
            lr_scheduler = {
                'scheduler': CosineWarmupScheduler(optimizer,
                                                   self.config.optim.warmup, self.config.training.n_iters),
                'interval': 'step',
            }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def get_optimizer(self, config, params):
        if config.optim.optimizer == 'Adam':
            return optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                              eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'AdamW':
            return optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.98),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'RAdam':
            return optim.RAdam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999),
                               eps=config.optim.eps, weight_decay=config.optim.weight_decay)
        else:
            raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported yet!')

    def on_save_checkpoint(self, checkpoint):
        checkpoint['model_ema'] = self.model_ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        self.model_ema.load_state_dict(checkpoint['model_ema'])


def main(args, config, try_resume):
    pl.seed_everything(config.seed)
    config.name = args.name
    data_path = os.path.join(args.data_root, args.version, 'train')

    # Initialize the PyTorch Lightning data module and model
    data_module = AMASSDataModule(config, args)
    data_module.setup(stage='fit')
    model = DPoserTrainer(config, args.bodymodel_path, data_path, N_POSES,
                          train_loader=data_module.train_dataloader(),
                          val_loader=data_module.val_dataloader(), )

    # Define logger and callbacks
    logger = TensorBoardLogger(f"logs/dposer/{config.dataset}", name=args.name)
    ckpt_dir = f"checkpoints/dposer/{config.dataset}/{args.name}"
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch:02d}-{step}-{val_mpjpe:.2f}',
                                          every_n_train_steps=config.training.save_freq,
                                          save_top_k=3, save_last=True, monitor='val_mpjpe', mode='min')
    model_logger = ModelSizeCallback()
    time_monitor = TimerCallback()
    lr_monitor = LearningRateMonitor()

    # Resume training
    resume_from_checkpoint = None
    if args.resume_ckpt is not None:
        resume_from_checkpoint = os.path.join(ckpt_dir, args.resume_ckpt)
        print('Resuming the training from {}'.format(resume_from_checkpoint))
    elif try_resume:
        available_ckpts = os.path.join(ckpt_dir, 'last.ckpt')
        if os.path.exists(available_ckpts):
            resume_from_checkpoint = os.path.realpath(available_ckpts)
            print('Resuming the training from {}'.format(resume_from_checkpoint))

    # Initialize the trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.devices,
        strategy='ddp',
        max_steps=config.training.n_iters,
        num_sanity_val_steps=5,
        val_check_interval=config.training.eval_freq,
        check_val_every_n_epoch=None,
        log_every_n_steps=config.training.log_freq,
        gradient_clip_val=config.optim.grad_clip,
        logger=logger,
        callbacks=[model_logger, time_monitor, lr_monitor, checkpoint_callback],
        benchmark=True,
        limit_val_batches=20,
    )

    # Train the model
    trainer.fit(model, ckpt_path=resume_from_checkpoint)


if __name__ == '__main__':
    args = parse_args(sys.argv)
    config = import_configs(args.config_path)
    # FIXME: there seems to be a bug in PyTorch Lightning while loading EMA parameters from resume.
    resume_training_if_possible = False
    main(args, config, resume_training_if_possible)
