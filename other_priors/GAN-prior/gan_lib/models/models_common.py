from easydict import EasyDict as edict
import torch

DEVICE_CPU = torch.device('cpu')

def get_models_by_config(cfg, gpus=[0], device0=DEVICE_CPU):
    assert len(cfg.MODELS) > 0, 'At least one network must be used!'
    models = edict()

    for model in cfg.MODELS: # "model" must be a name of a model architecture
        model_dict = cfg.MODELS[model]
        exec(f'from . import {model_dict.DIR}') # DIR - filename that keeps the architecture class
        params = model_dict.PARAMS if 'PARAMS' in model_dict else {}
        models[model] = eval(f'{model_dict.DIR}'+"."+f'{model_dict.ARCH}')(**params) 
        # ARCH - network class name

        if 'REQUIRES_GRAD' in model_dict and not model_dict.REQUIRES_GRAD:
            [param.requires_grad_(False) for param in models[model].parameters()]

        # wrap in torch.nn.Dataparallel
        models[model] = torch.nn.DataParallel(models[model], device_ids=gpus).to(device0)

    return models
