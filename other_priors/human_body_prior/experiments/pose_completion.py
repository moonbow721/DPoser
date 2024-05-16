import argparse
import os
import sys
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, '/data3/ljz24/projects/3d')
sys.path.insert(0, '/data3/ljz24/projects/3d/human_body_prior/src')
sys.path.insert(0, '/data3/ljz24/projects/3d/DPoser')
from os import path as osp
from lib.body_model.body_model import BodyModel
from lib.dataset.body.AMASS import AMASSDataset
from lib.dataset.body import Evaler
from lib.utils.misc import create_mask
from lib.body_model.visual import multiple_render

support_dir = '../support_data/dowloads'

device = 'cuda:0'


class SamplePose(object):
    def __init__(self, vposer, body_model, debug=False, device=None,
                 batch_size=1):
        self.debug = debug
        self.device = device
        self.VAE = vposer
        self.body_model = body_model
        self.betas = torch.zeros((batch_size, 10)).to(device=self.device)  # for visualization
        self.data_loss = nn.MSELoss(reduction='mean')

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'data': lambda cst, it: 10.0 * cst / (1 + it),
                       'vposer': lambda cst, it: 1000.0 * cst}
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def optimize(self, observation, mask, lr=0.5, iterations=2, steps_per_iter=100):
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

                encoding = self.VAE.encode(full_data).mean
                l2_norm_per_sample = torch.norm(encoding, p=2, dim=[1], keepdim=True)
                loss_dict['vposer'] = torch.mean(l2_norm_per_sample)
                loss_dict['data'] = self.data_loss(full_data * mask.float(), observation * mask.float())
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                # print(it*steps_per_iter+i, loss_dict['vposer'], tot_loss)
                tot_loss.backward()

                optimizer.step()

                full_data = torch.where(mask, observation, opti_variable)

        return full_data


def completion(part, hypo_num, sample):
    ### load the model
    from human_body_prior.tools.model_loader import load_vposer

    expr_dir = osp.join(support_dir, 'vposer_v1_0')
    vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
    vposer_pt = vposer_pt.to(device)

    test_dataset = AMASSDataset(root_path='/data3/ljz24/projects/3d/DPoser/body_data',
                                version='version1', subset='test', sample_interval=sample,)
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
    pose_sampler = SamplePose(vposer_pt, body_model=body_model, batch_size=batch_size, device=device)

    collected_results = []
    collected_dict = {}
    for idx, batch_data in enumerate(test_loader):
        vis = True if idx == 0 else False
        gts = batch_data['body_pose'].to(device, non_blocking=True)
        # mask, observation = create_mask(gts, part=part, observation_type='noise')

        all_hypos = []
        for i in range(hypo_num):
            mask, observation = create_mask(gts, part=part, observation_type='noise')
            completion = pose_sampler.optimize(observation, mask)  # [batch_size, 32]
            all_hypos.append(completion)
        all_hypos = torch.stack(all_hypos, dim=1)  # [batch_size, hypo, 32]

        evaler = Evaler(body_model=body_model, part=part)

        eval_results = evaler.multi_eval_bodys_all(all_hypos, gts)
        collected_results.append(eval_results)

    for single_process_results in collected_results:
        for key, value in single_process_results.items():
            if key not in collected_dict:
                collected_dict[key] = []
            collected_dict[key].extend(value)  # 合并数组

    print(f'results for {hypo_num} evals on {part}')
    for key, value in collected_dict.items():
        average_value = np.mean(np.array(value))
        print(f"The average of {key} is {average_value}")


def toy_completion(part, hypo_num, view, out_path=None):
    torch.manual_seed(42)

    ### load the model
    from human_body_prior.tools.model_loader import load_vposer

    expr_dir = osp.join(support_dir, 'vposer_v1_0')
    vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
    vposer_pt = vposer_pt.to(device)

    sample_num = 20
    # create Motion denoiser layer
    body_model = BodyModel(bm_path='/data3/ljz24/projects/3d/body_models/smplx/SMPLX_NEUTRAL.npz',
                           num_betas=10,
                           batch_size=sample_num,
                           model_type='smplx').to(device)
    pose_sampler = SamplePose(vposer_pt, body_model=body_model, batch_size=sample_num, device=device)

    file_path = '/data3/ljz24/projects/3d/DPoser/examples/toy_body_data.npz'
    data = np.load(file_path, allow_pickle=True)
    body_poses = data['pose_samples'][:sample_num]
    print(f'loaded axis pose data {body_poses.shape} from {file_path}')
    gts = torch.from_numpy(body_poses).to(device)
    mask, observation = create_mask(gts, part=part, observation_type='noise', model='body')

    all_hypos = []
    for i in range(hypo_num):
        completion = pose_sampler.optimize(observation, mask)  # [batch_size, 32]
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
    parser = argparse.ArgumentParser(
        description='VPoser completion.'
    )
    parser.add_argument('--outpath_folder', '-out', default='./vposer_completion/legs', type=str,
                        help='Path to output')
    parser.add_argument('--view', default='front', type=str, help='view direction')
    parser.add_argument('--part', default='legs', type=str, help='view direction')
    parser.add_argument('--sample', type=int, help='sample testset to reduce data for other tasks')
    args = parser.parse_args()

    # completion(part='trunk', hypo_num=10, sample=args.sample)
    toy_completion(part=args.part, hypo_num=10, view=args.view, out_path=args.outpath_folder)
    '''
    RUN:
    python my_completion.py --part legs
    '''
