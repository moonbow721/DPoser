import argparse
import os

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lib.body_model.utils import BodyPartIndices, BodySegIndices
from lib.dataset.body.AMASS import AMASSDataset

N_POSES = 21


class AMASSDataModule(pl.LightningDataModule):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

    def setup(self, stage=None):
        # Prepare data for training and validation
        if stage == 'fit' or stage is None:
            self.train_dataset = AMASSDataset(root_path=self.args.data_root,
                                              version=self.args.version,
                                              subset='train',
                                              sample_interval=self.args.sample,)

            self.val_dataset = AMASSDataset(root_path=self.args.data_root,
                                            version=self.args.version,
                                            subset='valid',
                                            sample_interval=100,)

        # Prepare data for testing
        if stage == 'test' or stage is None:
            self.test_dataset = AMASSDataset(root_path=self.args.data_root,
                                             version=self.args.version,
                                             subset='test',
                                             sample_interval=100,)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size,
                          num_workers=8, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=8, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.eval.batch_size,
                          num_workers=8, shuffle=False, drop_last=True)


# a simple eval process for completion task
class Evaler:
    def __init__(self, body_model, part=None):
        self.body_model = body_model
        self.part = part

        if self.part is not None:
            self.joint_idx = np.array(getattr(BodyPartIndices, self.part)) + 1  # skip pelvis
            self.vert_idx = np.array(getattr(BodySegIndices, self.part))
        else:
            self.joint_idx = slice(None)
            self.vert_idx = slice(None)

    def eval_bodys(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of body poses
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict for every sample [b,]
        '''
        sample_num = len(outs)
        eval_result = {'mpvpe': [], 'mpjpe': []}
        body_gt = self.body_model(body_pose=gts)
        body_out = self.body_model(body_pose=outs)

        for n in range(sample_num):
            # MPVPE from all vertices
            mesh_gt = body_gt.v.detach().cpu().numpy()[n, self.vert_idx]
            mesh_out = body_out.v.detach().cpu().numpy()[n, self.vert_idx]
            eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out - mesh_gt) ** 2, 1)).mean() * 1000)

            joint_gt_body = body_gt.Jtr.detach().cpu().numpy()[n, self.joint_idx]
            joint_out_body = body_out.Jtr.detach().cpu().numpy()[n, self.joint_idx]

            eval_result['mpjpe'].append(
                np.sqrt(np.sum((joint_out_body - joint_gt_body) ** 2, 1)).mean() * 1000)

        return eval_result

    def multi_eval_bodys(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_result = {f'mpvpe': [], f'mpjpe': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_result['mpvpe'].append(result['mpvpe'])
            eval_result['mpjpe'].append(result['mpjpe'])

        eval_result['mpvpe'] = np.min(eval_result['mpvpe'], axis=0)
        eval_result['mpjpe'] = np.min(eval_result['mpjpe'], axis=0)

        return eval_result

    def multi_eval_bodys_all(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict [b,]
        '''
        hypo_num = outs.shape[1]
        eval_collector = {f'mpvpe': [], f'mpjpe': []}
        eval_result = {f'mpvpe_best': [], f'mpjpe_best': [],
                       f'mpvpe_mean': [], f'mpjpe_mean': [],
                       f'mpvpe_std': [], f'mpjpe_std': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_collector['mpvpe'].append(result['mpvpe'])
            eval_collector['mpjpe'].append(result['mpjpe'])

        eval_result['mpvpe_best'] = np.min(eval_collector['mpvpe'], axis=0)
        eval_result['mpjpe_best'] = np.min(eval_collector['mpjpe'], axis=0)
        eval_result['mpvpe_mean'] = np.mean(eval_collector['mpvpe'], axis=0)
        eval_result['mpjpe_mean'] = np.mean(eval_collector['mpjpe'], axis=0)
        eval_result['mpvpe_std'] = np.std(eval_collector['mpvpe'], axis=0)
        eval_result['mpjpe_std'] = np.std(eval_collector['mpjpe'], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))

    def print_eval_result_all(self, eval_result):
        # complete this
        print('MPVPE best: %.2f mm' % np.mean(eval_result['mpvpe_best']))
        print('MPJPE best: %.2f mm' % np.mean(eval_result['mpjpe_best']))
        print('MPVPE mean: %.2f mm' % np.mean(eval_result['mpvpe_mean']))
        print('MPJPE mean: %.2f mm' % np.mean(eval_result['mpjpe_mean']))
        print('MPVPE std: %.2f mm' % np.mean(eval_result['mpvpe_std']))
        print('MPJPE std: %.2f mm' % np.mean(eval_result['mpjpe_std']))

    def print_multi_eval_result(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']))
        print(f'multihypo {hypo_num} MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
