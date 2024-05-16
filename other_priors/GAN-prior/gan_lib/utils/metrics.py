import os
import torch

from easydict import EasyDict as edict

def get_meters_by_config(cfg, losses_types):
    meters = edict()
    meters['train'] = edict()
    meters['valid'] = edict()

    losses_types = list(losses_types)

    #### Create "full" losses (it is useful if final loss is a combination of several other)
    losses_types.append('full')
    
    for mode in ['train', 'valid']:
        for loss_type in losses_types:
            meter_name = f'{loss_type}_{mode}'
            meters[mode][loss_type] = AvgMeter(meter_name=meter_name)
    return meters


class AvgMeter(object):
    """Computes and stores the average and current epoch values of losses or accuracies."""
    def __init__(self, meter_name='meter'):
        self.prev_vals = [] # averages for previous epochs
        self.meter_name = meter_name
        self.reset()

    def reset(self):
        '''resets the state of current epoch, saving new value as epoch ends'''
        self.cur_vals = [] # values for the current epoch, not necessarily all batches are processed
        self.cur_val = 0
        self.cur_avg = 0 # average over the number of inputs
        self.cur_sum = 0
        self.count = 0

    def update(self, cur_val, n=1):
        self.cur_vals.append(cur_val)
        self.cur_val = cur_val
        self.cur_sum += cur_val * n
        self.count += n
        self.cur_avg = self.cur_sum / self.count if self.count != 0 else 0

    def epochends(self): # will average current_vals and add this new value to the prev_vals
        self.prev_vals.append(self.cur_avg)
        self.reset()

    def save_state(self, output_dir):
        # by default saving is done for the format .pth
        dict2save = {
                    'meter_name' : self.meter_name,
                    'prev_vals' : self.prev_vals,
                    }
        metrics_dir = os.path.join(output_dir,'metrics')
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        torch.save(dict2save, os.path.join(metrics_dir, self.meter_name+'.pth'))
        # np.save(os.path.join(metrics_dir, self.meter_name+'.npy'), dict2save)

    def load_state(self, file, n_first_vals=None):
        dict2load = torch.load(file)
        # dict2load = np.load(file, allow_pickle=True).item()
        self.prev_vals = dict2load['prev_vals']
        if n_first_vals is not None:
            self.prev_vals = self.prev_vals[:n_first_vals]    

    def __repr__(self):
        prev_vals = self.prev_vals.__repr__()
        cur_vals = self.cur_vals.__repr__()
        return str(self.meter_name) + '\n' + 'prev_vals: ' + prev_vals + '\n' + 'cur_vals: ' + cur_vals + '\n'


