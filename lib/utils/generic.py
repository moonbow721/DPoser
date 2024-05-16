import importlib
import logging
import os
import time
from pathlib import Path
from collections import OrderedDict

import torch

from lib.algorithms.ema import ExponentialMovingAverage


def create_logger(cfg, phase='train', no_logger=False, folder_name=''):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET + '_' + cfg.DATASET.TEST_DATASET
    dataset = dataset.replace(':', '_')

    # cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

    if folder_name:
        final_output_dir = root_output_dir / dataset / f'{time_str}-{folder_name}'
    else:
        final_output_dir = root_output_dir / dataset / time_str

    # only get final output dir for distributed usage
    if no_logger:
        return None, str(final_output_dir), None

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head,
                        force=True)  # >= python 3.8
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / time_str
    # print('=> creating {}'.format(tensorboard_log_dir))
    # tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(final_output_dir)


def import_configs(config_path):
    module_name, function_name = config_path.rsplit('.', 1)
    config_module = importlib.import_module(module_name)
    get_config = getattr(config_module, function_name)
    config = get_config()
    return config


def find_npz_files(data_dir):
    npz_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.relpath(os.path.join(root, file), data_dir))
    return npz_files


def load_pl_weights(model, pl_weights, model_key='model'):
    state_dict = {k.replace(f"{model_key}.", ""): v for k, v in
                  pl_weights.items() if k.startswith(f"{model_key}.")}
    model.load_state_dict(state_dict)


def load_model(model, config, ckpt_path, device, is_ema=True):
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    # restore checkpoint
    if ckpt_path.endswith('.ckpt'):
        checkpoint = torch.load(ckpt_path, map_location=device)
        load_pl_weights(model, checkpoint['state_dict'])
        ema.load_state_dict(checkpoint['model_ema'])
        print(f"=> loaded checkpoint '{ckpt_path}' (step {checkpoint['global_step']})")
    else:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema'])
        print(f"=> loaded checkpoint '{ckpt_path}' (step {checkpoint['step']})")

    if is_ema:
        ema.copy_to(model.parameters())