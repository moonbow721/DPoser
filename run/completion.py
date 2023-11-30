import os

import math
import numpy as np
import torch
import torch.nn as nn
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader

from lib.dataset.EvaSampler import DistributedEvalSampler
from lib.utils.misc import create_mask, linear_interpolation

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.algorithms.advanced.model import ScoreModelFC
import torch.distributed as dist
import torch.multiprocessing as mp
from lib.algorithms.advanced import sde_lib, sampling
from lib.algorithms.ema import ExponentialMovingAverage
from lib.algorithms.advanced import utils as mutils
from lib.dataset.AMASS import AMASSDataset, N_POSES, Posenormalizer
from lib.body_model.body_model import BodyModel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='test diffusion model for completion on whole AMASS')

    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/axis-zscore-400k.pth')
    parser.add_argument('--dataset-folder', type=str, default='../data/AMASS/amass_processed',
                        help='the folder includes necessary normalizing parameters')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='path of SMPLX model')

    parser.add_argument('--hypo', type=int, default=1, help='number of hypotheses to sample')
    parser.add_argument('--part', type=str, default='left_leg', choices=['left_leg', 'right_leg', 'left_arm',
                                                                         'right_arm', 'trunk', 'hands',
                                                                         'legs', 'arms'])
    # optional
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--sample', type=int, help='sample testset to reduce data for other tasks')
    parser.add_argument('--batch_size', type=int, default=100, )
    parser.add_argument('--gpus', type=int, help='num gpus to inference parallel')
    parser.add_argument('--port', type=str, default='14600', help='master port of machines')

    args = parser.parse_args(argv[1:])

    return args


def get_dataloader(dataset, num_replicas=1, rank=0, batch_size=10000):
    sampler = DistributedEvalSampler(dataset,
                                     num_replicas=num_replicas,
                                     rank=rank,
                                     shuffle=False)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            sampler=sampler,
                            persistent_workers=False,
                            pin_memory=True,
                            drop_last=True)

    return dataloader


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


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

        if multi_denoise:   # not recommended
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

    def optimize(self, observation, mask, time_strategy='3', lr=0.1,
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
                if time_strategy == '1':
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

        opti_variable = observation * mask + opti_variable * (1.0 - mask)

        return opti_variable


def inference(rank, args, config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.gpus, args.port)

    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda", rank)

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
    else:
        raise NotImplementedError('unsupported model')
    model.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=None, model=model, ema=ema, step=0)

    # restore checkpoint
    map_location = {'cuda:0': 'cuda:%d' % rank}
    checkpoint = torch.load(args.ckpt_path, map_location=map_location)

    model.load_state_dict(checkpoint['model_state_dict'])
    ema.load_state_dict(checkpoint['ema'])
    state['step'] = checkpoint['step']
    print(f"=> loaded checkpoint '{args.ckpt_path}' (step {state['step']})")

    model.eval()
    ema.copy_to(model.parameters())

    # Setup SDEs
    if config.training.sde.lower() == 'vpsde':
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=args.steps)
    elif config.training.sde.lower() == 'subvpsde':
        sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=args.steps)
    elif config.training.sde.lower() == 'vesde':
        sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=args.steps)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Setup sampling functions
    compfn = DPoserComp(model, sde, config.training.continuous, batch_size=args.batch_size)
    Normalizer = Posenormalizer(data_path=f'{args.dataset_folder}/{args.version}/train',
                                normalize=config.data.normalize,
                                min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=device)

    test_dataset = AMASSDataset(root_path=args.dataset_folder,
                                version=args.version, subset='test', sample_interval=args.sample,
                                rot_rep=config.data.rot_rep, return_shape=False,
                                normalize=config.data.normalize, min_max=config.data.min_max)
    batch_size = args.batch_size
    test_loader = get_dataloader(test_dataset, num_replicas=args.gpus, rank=rank,
                                 batch_size=batch_size)
    body_model = BodyModel(bm_path=args.bodymodel_path,
                           num_betas=10,
                           batch_size=batch_size,
                           model_type='smplx').to(device)

    if rank == 0:
        print(f'total samples with reduction: {len(test_dataset)}')

    all_results = []

    for _, batch_data in enumerate(test_loader):
        poses = batch_data['poses'].to(device, non_blocking=True)
        mask, observation = create_mask(poses, part=args.part)

        multihypo_denoise = []
        for hypo in range(args.hypo):
            completion = compfn.optimize(observation, mask)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = Normalizer.offline_denormalize(multihypo_denoise, to_axis=True)
        gts = Normalizer.offline_denormalize(poses, to_axis=True)

        from lib.dataset.AMASS import Evaler
        evaler = Evaler(body_model=body_model, part=args.part)
        eval_results = evaler.multi_eval_bodys(preds, gts)  # [batch_size, ]
        all_results.append(eval_results)

    # collect data from other process
    print(f'rank[{rank}] subset len: {len(all_results)}')

    results_collection = [None for _ in range(args.gpus)]
    dist.gather_object(
        all_results,
        results_collection if rank == 0 else None,
        dst=0
    )

    if rank == 0:
        collected_results = np.concatenate(results_collection, axis=0)  # [batch_num,], every batch result is a dict
        collected_dict = {}

        # gather and settle results from all ranks
        for single_process_results in collected_results:
            for key, value in single_process_results.items():
                if key not in collected_dict:
                    collected_dict[key] = []
                collected_dict[key].extend(value)

        # compute the mean value
        for key, value in collected_dict.items():
            average_value = np.mean(np.array(value))
            print(f"The average of {key} is {average_value}")

    cleanup()


def main(args):
    # mp.freeze_support()
    mp.set_start_method('spawn')

    config = FLAGS.config

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=inference, args=(rank, args, config))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)