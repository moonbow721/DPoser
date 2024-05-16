import os
from types import SimpleNamespace

import numpy as np
import torch
from absl import app
from absl import flags
from absl.flags import argparse_flags
from ml_collections.config_flags import config_flags
from torch.utils.data import DataLoader

from lib.algorithms.completion import DPoserComp
from lib.dataset.EvaSampler import DistributedEvalSampler
from lib.utils.generic import load_pl_weights, load_model
from lib.utils.misc import create_mask

try:
    from tensorboardX import SummaryWriter
except ImportError as e:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        print('Tensorboard is not Installed')

from lib.algorithms.advanced.model import create_model
import torch.distributed as dist
import torch.multiprocessing as mp
from lib.algorithms.advanced import sde_lib, sampling
from lib.dataset.body.AMASS import AMASSDataset
from lib.dataset.body import N_POSES, Evaler
from lib.dataset.utils import Posenormalizer
from lib.body_model.body_model import BodyModel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='test diffusion model for completion on whole AMASS')

    parser.add_argument('--ckpt-path', type=str, default='./pretrained_models/amass/BaseMLP/epoch=36-step=150000-val_mpjpe=38.17.ckpt')
    parser.add_argument('--dataset-folder', type=str,
                        default='../data/human/Bodydataset/amass_processed',
                        help='the folder includes necessary normalizing parameters')
    parser.add_argument('--version', type=str, default='version1', help='dataset version')
    parser.add_argument('--bodymodel-path', type=str, default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='path of SMPLX model')

    parser.add_argument('--hypo', type=int, default=1, help='number of hypotheses to sample')
    parser.add_argument('--part', type=str, default='left_leg', choices=['left_leg', 'right_leg', 'left_arm',
                                                                         'right_arm', 'trunk', 'hands',
                                                                         'legs', 'arms'])
    # optional
    parser.add_argument('--mode', default='DPoser', choices=['DPoser', 'ScoreSDE', 'MCG', 'DPS'])
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


def inference(rank, args, config):
    print(f"Running DDP on rank {rank}.")
    setup(rank, args.gpus, args.port)

    ## Load the pre-trained checkpoint from disk.
    device = torch.device("cuda", rank)
    POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
    model = create_model(config.model, N_POSES, POSE_DIM)
    model.to(device)
    model.eval()
    load_model(model, config, args.ckpt_path, device, is_ema=True)

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
    comp_fn = DPoserComp(model, sde, config.training.continuous, batch_size=args.batch_size)
    Normalizer = Posenormalizer(
        data_path=os.path.join(args.dataset_folder, args.version, 'train'),
        normalize=config.data.normalize, min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=device)

    # Perform completion baselines (ScoreSDE, MCG, DPS)
    task_args = SimpleNamespace(task=None)
    if args.mode == 'DPS':
        task_args.task, inverse_solver = 'default', 'BP'
    elif args.mode == 'MCG':
        task_args.task, inverse_solver = 'completion', 'BP'
    elif args.mode == 'ScoreSDE':
        task_args.task, inverse_solver = 'completion', None
    else:  # plain generation sampler
        task_args.task, inverse_solver = 'default', None
    comp_sampler = sampling.get_sampling_fn(config, sde, (args.batch_size, N_POSES * POSE_DIM),
                                            lambda x: x, 1e-3, device=device, inverse_solver=inverse_solver)

    test_dataset = AMASSDataset(root_path=args.dataset_folder,
                                version=args.version, subset='test', sample_interval=args.sample,)
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
    evaler = Evaler(body_model=body_model, part=args.part)

    for _, batch_data in enumerate(test_loader):
        poses = batch_data['body_pose'].to(device, non_blocking=True)
        poses = Normalizer.offline_normalize(poses, from_axis=True)
        mask, observation = create_mask(poses, part=args.part, model='body')

        multihypo_denoise = []
        for hypo in range(args.hypo):
            if args.mode == 'DPoser':
                completion = comp_fn.optimize(observation, mask)
            else:
                _, completion = comp_sampler(model, observation=observation, mask=mask, args=task_args)
            multihypo_denoise.append(completion)
        multihypo_denoise = torch.stack(multihypo_denoise, dim=1)

        preds = Normalizer.offline_denormalize(multihypo_denoise, to_axis=True)
        gts = Normalizer.offline_denormalize(poses, to_axis=True)

        eval_results = evaler.multi_eval_bodys_all(preds, gts)  # [batch_size, ]
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
