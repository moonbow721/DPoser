"""positional encoding"""
import torch
from torch.autograd import grad

class PosEncoder():
    def __init__(self, number_frequencies, include_identity):
        freq_bands = torch.pow(2, torch.linspace(0., number_frequencies - 1, number_frequencies))
        self.embed_fns = []
        self.output_dim = 0
        self.number_frequencies = number_frequencies
        self.include_identity = include_identity
        if include_identity:
            self.embed_fns.append(lambda x: x)
            self.output_dim += 1
        if number_frequencies > 0:
            for freq in freq_bands:
                for periodic_fn in [torch.sin, torch.cos]:
                    self.embed_fns.append(lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq))
                    self.output_dim += 1

    def encode(self, coordinate):
        return torch.cat([fn(coordinate) for fn in self.embed_fns], -1)


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    # points_grad = grad(
    #     outputs=outputs,
    #     inputs=inputs,
    #     grad_outputs=d_points,
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True)[0][:, -3:]   #todo: why was this hardcoded
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def get_parent_mapping(model_type):
    """this is mapping without root joint"""
    smpl_mappings = [ -1, -1, -1, 1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
    if model_type != 'smpl':
        print("Model hierarchy not defined.....")
        return None
    return smpl_mappings

def get_parent_mapping_old(model_type):
    smplh_mappings = [
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25,
        26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50
    ]

    if model_type == 'smpl':
        smpl_mappings = smplh_mappings[:22] + [smplh_mappings[25]] + [smplh_mappings[40]]
        return smpl_mappings
    return smplh_mappings