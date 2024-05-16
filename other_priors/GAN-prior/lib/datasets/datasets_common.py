from easydict import EasyDict as edict
from copy import copy

import torchvision.transforms as transforms
import torch
import warnings

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1,-1,1,1)
        self.std = torch.tensor(std).view(1,-1,1,1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor * self.std + self.mean
        tensor = (tensor * 255).type(th.int16)
        return tensor

UNNORMALIZE = UnNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
TRANSFORMS = {
    'W/ NORM'  : transforms.Compose([transforms.ToTensor(), NORMALIZE]),
    'W/O NORM' : transforms.Compose([transforms.ToTensor()])
    }

def get_datasets_by_config(cfg):
    datasets = edict()
    
    if len(cfg.DATASETS) == 0:
        warnings.warn("No datasets were specified", RuntimeWarning)

    for dset in cfg.DATASETS:
        dset_dict = cfg.DATASETS[dset]

        if 'CONCAT' in dset_dict and dset_dict.CONCAT:
            assert 'DSETS' in dset_dict, 'If one dset is Concat of some other, the list of such must be specified.'
            datasets[dset] = torch.utils.data.ConcatDataset([datasets[ds] for ds in dset_dict.DSETS])

        elif 'SMART_CONCAT' in dset_dict and dset_dict.SMART_CONCAT:
            assert 'DSETS' in dset_dict, 'If one dset is SmartConcat of some other, ' \
                                         'the list of such must be specified.'
            from .smart_concat import SmartConcatDataset
            datasets[dset] = SmartConcatDataset([datasets[ds] for ds in dset_dict.DSETS], dset_dict.KEYS)
        else:   
            exec(f'from . import {dset_dict.DIR}')
            params = dset_dict.PARAMS if 'PARAMS' in dset_dict else {}
            datasets[dset] = eval(f'{dset_dict.DIR}'+"."+f'{dset_dict.NAME}')(**params)

    return datasets


def get_dataloaders_by_config(cfg, datasets, num_gpus=1):

    dataloaders = edict()

    for dset in cfg.DATALOAD:
        
        # TODO what if name is different? as for mixed dl
        # assert dset in datasets, '"dataloader" name must coincide with the name of dataset! Check <exp>.yaml file' 
        dl_dict = cfg.DATALOAD[dset]

        if dset == 'multidl': ### uses custom MultiDataLoaderForIterations class 
            ### TODO 'dls' key and 'NUM_ITERATIONS_PER_EPOCH' must be in the dictionary if one uses multidl 
            ### TODO all dls specified in dictionary must be presented before initializing multidl
            exec(f'from .dataloaders_custom import MultiDataLoaderForIterations')
            dls = [dataloaders[dl] for dl in dl_dict.dls] 
            dl = eval('MultiDataLoaderForIterations')(dls, dl_dict.NUM_ITERATIONS_PER_EPOCH)
            dataloaders[dset] = dl
            continue

        params = copy(dl_dict.PARAMS) if 'PARAMS' in dl_dict else {}
        params['batch_size'] = params.batch_size * max(num_gpus, 1) if 'batch_size' in params \
                               else 32 # custom default
        if 'pin_memory' not in params:
            params['pin_memory'] = True # custom default
        if 'num_workers' not in params:
            params['num_workers'] = 10 # custom default TODO
        if 'collate_fn' in params:
            collate_fn_dict = params.collate_fn
            exec(f'from . import {collate_fn_dict.DIR}')
            collate_fn = eval(f'{collate_fn_dict.DIR}'+"."+f'{collate_fn_dict.NAME}')
            params['collate_fn'] = collate_fn

        ### Warning! default values of DataLoader class are:
        ### batchsize == 1; 
        ### shuffle == False; 
        ### pin_memory == False. These parameters should be specified manually in the <exp>.yaml file

        ### name of dset and dl coincide    
        dl = torch.utils.data.DataLoader(datasets[dset], **params)
        
        if 'NUM_ITERATIONS_PER_EPOCH' in dl_dict: ### uses custom DataLoaderForIterations class
            exec(f'from .dataloaders_custom import DataLoaderForIterations')
            dl = eval('DataLoaderForIterations')(dl, dl_dict.NUM_ITERATIONS_PER_EPOCH)

        dataloaders[dset] = dl
    

    return dataloaders