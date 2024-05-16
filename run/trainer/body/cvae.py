import argparse
import os
import os.path as osp
import random
import sys

import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim

from lib.algorithms.vposer.losses import geodesic_loss_R
from lib.algorithms.vposer.model import CVPoser, CVPoser2
from lib.body_model.body_model import BodyModel
from lib.dataset.body import AMASSDataModule, N_POSES
from lib.utils.callbacks import ModelSizeCallback, TimerCallback
from lib.utils.generic import import_configs
from lib.utils.misc import create_mask
from lib.utils.schedulers import CosineWarmupScheduler
from lib.utils.transforms import axis_angle_to_mat3x3


def parse_args(argv):
    parser = argparse.ArgumentParser(description='train conditional vae model for completion')
    parser.add_argument('--config-path', '-c', type=str,
                        default='subprior.configs.optim.set1.get_config',
                        help='config files to build CVPoser')
    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX for visualization')
    parser.add_argument('--resume-ckpt', '-r', type=str, help='resume training')
    parser.add_argument('--data-root', type=str,
                        default='../data/human/Bodydataset/amass_processed', help='dataset root')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--shape', type=bool, default=False, help='handle human shapes')
    parser.add_argument('--sample', type=int, help='sample trainset to reduce data')
    parser.add_argument('--vposer', type=str, default='v1', choices=['v1', 'v2'])
    parser.add_argument('--name', type=str, default='default', help='name of checkpoint folder')

    args = parser.parse_args(argv[1:])

    return args


class VPoserTrainer(LightningModule):
    def __init__(self, config, vposer_version,
                 bodymodel_path='',
                 data_path='',
                 N_POSES=21,
                 train_loader=None,
                 val_loader=None, ):
        super(VPoserTrainer, self).__init__()
        self.config = config
        self.bodymodel_path = bodymodel_path
        self.data_path = data_path
        self.N_POSES = N_POSES
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_hyperparameters(ignore=['train_loader', 'val_loader'])

        self.POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
        if vposer_version == 'v1':
            self.model = CVPoser(config.model, N_POSES, self.POSE_DIM)
        else:
            self.model = CVPoser2(config.model, N_POSES, self.POSE_DIM)
        self.bm_train = BodyModel(bm_path=self.bodymodel_path,
                                  num_betas=10,
                                  batch_size=config.training.batch_size,
                                  model_type='smplx')
        for param in self.bm_train.parameters():
            param.requires_grad = False

        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.geodesic_loss = geodesic_loss_R(reduction='mean')
        self.loss_weight = config.loss
        self.keep_extra_loss_terms_until_epoch = config.loss.keep_extra_loss_terms_until_epoch
        self.all_parts = ['trunk', 'left_leg', 'right_leg', 'left_arm', 'right_arm', 'hands', 'legs', 'arms']

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        part = random.choice(self.all_parts)
        mask, observation = create_mask(batch['body_pose'], part=part, model='body')
        dict_reconstruction = self.model(batch['body_pose'], observation)
        loss = self._compute_loss(batch, dict_reconstruction)
        train_loss = loss['weighted_loss']['loss_total']
        self.log('train_loss', train_loss, prog_bar=True, )
        return train_loss

    def validation_step(self, batch, batch_idx):
        part = random.choice(self.all_parts)
        mask, observation = create_mask(batch['body_pose'], part=part, model='body')
        dict_reconstruction = self.model(batch['body_pose'], observation)
        loss = self._compute_loss(batch, dict_reconstruction)
        val_loss = loss['unweighted_loss']['loss_total']
        self.log('val_loss', val_loss, prog_bar=True, sync_dist=True)

        return val_loss

    def _compute_loss(self, original_data, reconstructed_data):
        batch_size, latent_dim = reconstructed_data['poZ_body_mean'].shape

        # Loss weights
        loss_kl_weight = self.loss_weight.w_kl
        loss_rec_weight = self.loss_weight.w_recon
        loss_matrot_weight = self.loss_weight.w_mat
        loss_jtr_weight = self.loss_weight.w_jtr

        # KL loss
        q_z = reconstructed_data['q_z']
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((batch_size, latent_dim), device=self.device),
            scale=torch.ones((batch_size, latent_dim), device=self.device))

        kl_loss = loss_kl_weight * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=1))

        # Reconstruction loss
        with torch.no_grad():
            bm_orig = self.bm_train(body_pose=original_data['body_pose'])
        bm_rec = self.bm_train(body_pose=reconstructed_data['pose_body'].reshape(batch_size, -1))

        mesh_rec_loss = loss_rec_weight * self.l1_loss(bm_rec.v, bm_orig.v)

        weighted_loss_dict = {
            'loss_kl': kl_loss,
            'loss_mesh_rec': mesh_rec_loss
        }

        # Extra losses
        if self.current_epoch < self.keep_extra_loss_terms_until_epoch:
            matrot_loss = loss_matrot_weight * self.geodesic_loss(reconstructed_data['pose_body_matrot'].view(-1, 3, 3),
                                                                  axis_angle_to_mat3x3(
                                                                      original_data['body_pose'].view(-1, 3)))
            jtr_loss = loss_jtr_weight * self.l1_loss(bm_rec.Jtr, bm_orig.Jtr)
            weighted_loss_dict.update({'matrot': matrot_loss, 'jtr': jtr_loss})

        total_weighted_loss = torch.stack(list(weighted_loss_dict.values())).sum()

        weighted_loss_dict['loss_total'] = total_weighted_loss

        # Unweighted losses
        v2v_loss = torch.sqrt(torch.pow(bm_rec.v - bm_orig.v, 2).sum(-1)).mean()
        total_unweighted_loss = v2v_loss

        unweighted_loss_dict = {
            'v2v': v2v_loss,
            'loss_total': total_unweighted_loss
        }

        return {'weighted_loss': weighted_loss_dict, 'unweighted_loss': unweighted_loss_dict}

    def configure_optimizers(self):
        gen_params = [a[1] for a in self.model.named_parameters() if a[1].requires_grad]
        optimizer = self.get_optimizer(self.config, gen_params)
        if self.config.optim.warmup > 0:
            schedulers = {
                'scheduler': CosineWarmupScheduler(optimizer,
                                                   self.config.optim.warmup,
                                                   self.config.training.num_epochs*len(self.val_dataloader())),
                'interval': 'step',
            }
        else:
            schedulers = [
                {
                    'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=5, verbose=True),
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                },
            ]
        return [optimizer], schedulers

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


def main(args, config, try_resume=False):
    pl.seed_everything(config.seed)
    config.name = args.name
    data_path = os.path.join(args.data_root, args.version, 'train')

    # Initialize the PyTorch Lightning data module and model
    data_module = AMASSDataModule(config, args)
    data_module.setup(stage='fit')
    model = VPoserTrainer(config, args.vposer, args.bodymodel_path, data_path, N_POSES,
                          train_loader=data_module.train_dataloader(),
                          val_loader=data_module.val_dataloader(), )

    logger = TensorBoardLogger(f"logs/cvposer/{config.dataset}", name=args.name)
    model_logger = ModelSizeCallback()
    time_monitor = TimerCallback()
    lr_monitor = LearningRateMonitor()

    ckpt_dir = f"checkpoints/cvposer/{config.dataset}/{args.name}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k=3, monitor='val_loss', save_last=True, verbose=True, mode='min', )

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10, verbose=True, mode='min')

    resume_from_checkpoint = None
    if args.resume_ckpt is not None:
        resume_from_checkpoint = osp.join(ckpt_dir, args.resume_ckpt)
        print('Resuming the training from {}'.format(resume_from_checkpoint))
    elif try_resume:
        available_ckpts = osp.join(ckpt_dir, 'last.ckpt')
        if osp.exists(available_ckpts):
            resume_from_checkpoint = os.path.realpath(available_ckpts)
            print('Resuming the training from {}'.format(resume_from_checkpoint))

    trainer = pl.Trainer(devices=config.devices,
                         callbacks=[model_logger, time_monitor, lr_monitor, early_stop_callback, checkpoint_callback],
                         max_epochs=config.training.num_epochs,
                         logger=logger,
                         strategy='ddp',
                         )

    trainer.fit(model, ckpt_path=resume_from_checkpoint)


if __name__ == '__main__':
    args = parse_args(sys.argv)
    config = import_configs(args.config_path)
    resume_training_if_possible = True
    main(args, config, resume_training_if_possible)