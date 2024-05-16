from easydict import EasyDict as edict

import torch

class Optimizer():
    def __init__(self, optimizers): # "optimizers" is supposed to be dict-like object 
        self.opts = optimizers

    def load_state(self, ckpt):
        for opt_key in self.opts:
            self.opts[opt_key].load_state_dict(ckpt[opt_key+'_state_dict_optim'])

    def zero_grad(self):
        for model_name in self.opts:
            self.opts[model_name].zero_grad()

    def step(self):
        for model_name in self.opts:
            self.opts[model_name].step()

    def save_state(self):
        state = {}
        for opt_key in self.opts:
            opt_name = opt_key + '_state_dict_optim'
            state[opt_name] = self.opts[opt_key].state_dict()
        return state

    def __repr__(self):
        return 'optimizers: \n' + self.opts.__repr__()


def get_optim_by_config(cfg, models):
    optimizers = edict()

    for optim in cfg.OPTIM: 
        assert optim in models, '"optim" name must coincide with the name of network! Check <exp>.yaml file'
        optim_dict = cfg.OPTIM[optim]
        exec(f'from torch.optim import {optim_dict.NAME}')
        params = optim_dict.PARAMS if 'PARAMS' in optim_dict else {}
        optimizers[optim] = eval(f'{optim_dict.NAME}')(models[optim].parameters(), **params)

    return Optimizer(optimizers)


###############################################################################################################


class Scheduler():
    def __init__(self, sched): # "sched" is supposed to be dict-like object
        self.sched = sched
        self.new_lrs = {model_name : None for model_name in self.sched}
        self.sched_finished = {model_name : False for model_name in self.sched}
        self.do_scheduling = len(self.sched) > 0

    def step(self, metrics=None):
        self.new_lrs = {model_name : None for model_name in self.sched}

        for model_name, sched in self.sched.items():
            if metrics is not None and sched.__class__.__name__ == 'ReduceLROnPlateau':
                old_lr = sched.optimizer.param_groups[0]['lr']
                sched.step(metrics)

                new_lr = sched.optimizer.param_groups[0]['lr']
                if old_lr - new_lr > sched.eps:
                    self.new_lrs[model_name] = new_lr

                    if abs(new_lr - sched.min_lrs[0]) < sched.eps:
                        self.sched_finished[model_name] = True
            else:
                sched.step()

    def load_state(self, ckpt):
        for sched_key in self.sched:
            self.sched[sched_key].load_state_dict(ckpt[sched_key+'_state_dict_sched'])

    def save_state(self):
        state = {}
        for sched_key in self.sched:
            sched_name = sched_key + '_state_dict_sched'
            state[sched_name] = self.sched[sched_key].state_dict()
        return state

    def __repr__(self):
        return 'schedulers: \n' + self.sched.__repr__()


def get_sched_by_config(cfg, optim):
    schedulers = edict()
    
    for sched in cfg.SCHED: 
        assert sched in optim.opts, \
        '"sched" name must coincide with the name of the optim! Check <exp>.yaml file'
        sched_dict = cfg.SCHED[sched]
        exec(f'from torch.optim.lr_scheduler import {sched_dict.NAME}')
        params = sched_dict.PARAMS if 'PARAMS' in sched_dict else {}
        schedulers[sched] = eval(f'{sched_dict.NAME}')(optim.opts[sched], **params)

    return Scheduler(schedulers)