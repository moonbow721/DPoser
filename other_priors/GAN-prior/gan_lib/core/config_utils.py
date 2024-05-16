import yaml

from easydict import EasyDict as edict
from copy import deepcopy

BASE_CONFIG = edict()

BASE_CONFIG.CONFIG_FILENAME = ''
BASE_CONFIG.OUTPUT_DIR = None
BASE_CONFIG.GPUS = ''
BASE_CONFIG.PRINT_FREQ = 100 # how often to print the results

### ### Cudnn related params
### BASE_CONFIG.CUDNN = edict()
### BASE_CONFIG.CUDNN.BENCHMARK = True
### BASE_CONFIG.CUDNN.DETERMINISTIC = False
### BASE_CONFIG.CUDNN.ENABLED = True

BASE_CONFIG.MODELS = edict()

BASE_CONFIG.LOSSES = edict()
BASE_CONFIG.OPTIM = edict()
BASE_CONFIG.SCHED = edict()

BASE_CONFIG.DATASETS = edict()
BASE_CONFIG.DATALOAD = edict()

BASE_CONFIG.PROCEDURE = ''
BASE_CONFIG.EXP_NAME = None

# training specifications
BASE_CONFIG.TRAINING = edict()
BASE_CONFIG.TRAINING.END_EPOCH = 100


def get_config(exp_config):
    config_filename = ''
    if isinstance(exp_config, str):
        config_filename = exp_config
        with open(exp_config) as f:
            exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
    elif not isinstance(exp_config, edict):
        raise ValueError('"exp_config" input must be of type either PATH (to the yaml file) or EASYDICT!')

    new_config = update_config(exp_config)
    new_config.CONFIG_FILENAME = config_filename # path to the config file relative to the launching point 
    return new_config

def update_config(exp_config, old_config=BASE_CONFIG):
    new_config = deepcopy(old_config)
    for k, v in exp_config.items():
        if k not in new_config:
            new_config[k] = v
        elif isinstance(v, dict): 
            # if key "k" in "new_config" dict, but we'd like to update it with "exp_config" value
            _update_dict(v, new_config[k])
        else:
            new_config[k] = v

    return new_config

def _update_dict(v, dict_to_update):
    for vk, vv in v.items():
        if vk not in dict_to_update:
            dict_to_update[vk] = vv
        elif isinstance(vv, dict):
            _update_dict(vv, dict_to_update[vk])
        else:
            dict_to_update[vk] = vv



if __name__ == '__main__':
    import sys
    exp_name = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = get_config(exp_name) if exp_name is not None else BASE_CONFIG
    for k in cfg.keys():
        print(k, cfg[k], sep=': ', end='\n')



