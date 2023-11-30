import importlib
import logging
import time
from pathlib import Path


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
