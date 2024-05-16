import torch
import numpy as np

import json
from os import path as osp

DATASETS = ['ACCAD', 'BMLhandball', 'BMLmovi', 'BMLrub', 'CMU', 'DFaust67', 'DanceDB', 'EKUT', 'EyesJapanDataset', 'HUMAN4D', 'HumanEva', 'KIT', 'MPIHDM05', 'MPILimits', 'MPImosh', 'SFU', 'SSMsynced', 'TCDhandMocap', 'TotalCapture', 'Transitionsmocap']

# N - number of samples per array in sequence
KEYS = [
    # 'gender', # 'male', 'female'
    # 'mocap_framerate', # int, e.g. 60., 100., 120.
    'trans', # (N, 3)
    'poses', # (N, 156)
    'betas', # (16,)
    'dmpls', # (N, 8)
    ]


ORIGINAL_AMASS_SPLITS = {
    'valid': ['HumanEva', 'MPIHDM05', 'SFU', 'MPImosh'], # 1'056'110 samples
    'test': ['Transitionsmocap', 'SSMsynced'], # 94'145
    'train': ['CMU', 'MPILimits', 'TotalCapture', 'EyesJapanDataset', 'KIT', 'BMLmovi', 'BMLrub', 'BMLhandball', 'EKUT', 'TCDhandMocap', 'ACCAD'] # 13'430'710 samples
}

hack_dataset_path = '/data3/ljz24/projects/3d/data/human/Bodydataset/amass_processed/version1'

class AMASS(torch.utils.data.Dataset):
    def __init__(self, 
                datasets=[], 
                num_betas=10, # 16 is max
                num_joints=21,
                shuffle=True,
                path_to_dataset=None,
                mode=None # 'train', 'valid', 'test' - as in VPoser training 
                        # https://github.com/nghorbani/human_body_prior/tree/1936f38aec4bb959f6a8bf4ed304b6aafb42fa30/human_body_prior/data
                ):

        # get all datasets if input datasets is empty
        self.datasets = DATASETS if datasets == [] else datasets

        if mode is not None:
            self.datasets = ORIGINAL_AMASS_SPLITS[mode]

        if path_to_dataset is None:
            raise ValueError
        self.path_to_dataset = path_to_dataset # path to the folder that contains AMASS subdatasets

        self.num_betas = num_betas
        self.num_joints = num_joints

        # self.load_datasets()
        # self.preproc_raw_data()

        self.load_hack_datasets()

    def load_hack_datasets(self, mode='train'):
        self.pose = torch.load(osp.join(hack_dataset_path, mode, 'pose_body.pt'))
        self.betas = torch.load(osp.join(hack_dataset_path, mode, 'betas.pt'))


    # def load_datasets(self):
    #     pose_arr, betas_arr  = [], []
    #     for dset_name in self.datasets:
    #         pose = torch.load(osp.join(self.path_to_dataset, dset_name, 'pose.pt'))
    #         betas = torch.load(osp.join(self.path_to_dataset, dset_name, 'betas.pt'))
    #
    #         pose_arr.append(pose)
    #         betas_arr.append(betas)
    #
    #     self.pose = torch.cat(pose_arr, dim=0)
    #     self.betas = torch.cat(betas_arr, dim=0)
    #
    # def preproc_raw_data(self):
    #     # keep only first "num_betas" betas components
    #     self.betas = self.betas[:,:self.num_betas]
    #
    #     # select SMPL-H body part from "pose" (21 joints)
    #     self.pose = self.pose[:,3:66]
    #
    #     if self.num_joints == 23:
    #         # add zeros for last two angles (as in BodyModel in VPoser)
    #         self.pose = torch.cat((self.pose, torch.zeros(self.pose.size(0), 6)), dim=1)

    def __len__(self):
        return len(self.pose)


    def __getitem__(self, idx):

        # pose, shape, meta = self.get_data(idx)
        # shape = torch.Tensor(shape)
        # global_orient = torch.Tensor(pose[:3]) # controls the global root orientation
        # pose_body = torch.Tensor(pose[3:66]) # controls the body
        # pose_hand = torch.Tensor(pose[66:]) # controls the finger articulation

        pose = self.pose[idx]
        betas = self.betas[idx]
        smpl_vec = torch.cat((pose, betas), dim=0)
        d = dict(pose=pose, betas=betas, smpl_vec=smpl_vec)
        return d
    


if __name__ == '__main__':

    ds = AMASS(datasets=['CMU', 'MPILimits'], num_betas=10, path_to_dataset='./data/amass/')

    seed = 128
    torch.manual_seed(seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
    import os
    os.makedirs(f'./AMASS_TESTS/seed{seed}/', exist_ok=True)
    for sample in dl:
        print(sample['pose'].shape, sample['betas'].shape)
        # torch.save(sample, f'./AMASS_TESTS/seed{seed}/AMASS_SAMPLE_seed{seed}.pth')
        break


### for AMASS SMPL-X


# DATASETS = ['ACCAD', 'BMLrub', 'CMU', 'DFaust', 'EKUT', 'Eyes_Japan_Dataset', 'HDM05', 'HumanEva', 'KIT', 'MoSh', 'PosePrior', 'SFU', 'SSM', 'TCDHands', 'TotalCapture', 'Transitions']

# # N - number of samples per array in sequence
# KEYS = [
#     'gender', # 'neutral' 
#     'surface_model_type', # 'smplx'
#     'mocap_frame_rate', # int, e.g. 120 
#     'mocap_time_length', # float, e.g. 4.36667 
#     'markers_latent', # (41, 3), (53, 3), (67, 3), (89, 3), ...
#     'latent_labels', # list of labels (different everywhere)
#     'trans', # N x 3 
#     'poses', # N x 165
#     'betas', # (16,)
#     'num_betas', # 16
#     'root_orient', # N x 3
#     'pose_body', # N x 63
#     'pose_hand', # N x 90
#     'pose_jaw', # N x 3
#     'pose_eye' # N x 6
#     ]


### Total num of samples : 16'258'989
### Total num of sequences : 11'394