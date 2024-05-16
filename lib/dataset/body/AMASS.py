import os

import torch
from torch.utils.data import DataLoader


class AMASSDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, version='version0', subset='train', sample_interval=None,):

        self.root_path = root_path
        self.version = version
        assert subset in ['train', 'valid', 'test']
        self.subset = subset
        self.sample_interval = sample_interval

        self.global_orients, self.body_poses = self.read_data()

        if self.sample_interval:
            self._sample(sample_interval)

        self.real_data_len = len(self.body_poses)

    def __getitem__(self, idx):
        """
        Return:
            [21, 3] or [21, 6] for poses including body and root orient
            [10] for shapes (betas)  [Optimal]
        """
        global_orient = self.global_orients[idx % self.real_data_len]
        body_pose = self.body_poses[idx % self.real_data_len]
        data_dict = {'global_orient': global_orient, 'body_pose': body_pose}

        return data_dict

    def __len__(self, ):
        return len(self.body_poses)

    def _sample(self, sample_interval):
        print(f'Class AMASSDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.global_orients = self.global_orients[::sample_interval]
        self.body_poses = self.body_poses[::sample_interval]

    def read_data(self):
        data_path = os.path.join(self.root_path, self.version, self.subset)
        global_orient = torch.load(os.path.join(data_path, 'root_orient.pt'))
        body_pose = torch.load(os.path.join(data_path, 'pose_body.pt'))

        return global_orient, body_pose

