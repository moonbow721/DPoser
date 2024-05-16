"""Data preparation code for PoseNDF training"""

from __future__ import division

import os
import numpy as np
import pytorch3d
import faiss
import ipdb
from data_splits import amass_splits

from torch.utils.data import Dataset
import os
import numpy as np
import torch
import ipdb
from pytorch3d.transforms import axis_angle_to_quaternion
import glob

from smplx import SMPL, SMPLH, SMPLX

def quat_doublecover(pose_in, samples=2000):
    n, k = pose_in.shape[0], pose_in.shape[1]
    pose_in = pose_in.reshape(n * k, 4)
    indices = np.random.randint(0, len(pose_in), samples)
    pose_in[indices] = pose_in[indices] * (-1)
    return pose_in.reshape(n, k, 4)

def quat_flip(pose_in):
    is_neg = pose_in[:,:,0] <0
    pose_in[np.where(is_neg)] = (-1)*pose_in[np.where(is_neg)]
    return pose_in, np.sum(is_neg)



def axis_angle_to_quaternion_np(pose_in):
    pose_pt = torch.from_numpy(pose_in)
    pose_out = axis_angle_to_quaternion(pose_pt).detach().numpy()

    return pose_out


class PoseData(Dataset):

    def __init__(self,data_path, mode='query', batch_size=1, num_workers=12, num_samples=128, runs=1000):

        self.path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.sigma = np.array([0.01, 0.05, 0.1, 0.25, 0.5])
        self.sample_distribution =  np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.num_sample_points = num_samples
        # self.all_files = [os.path.join(self.path, '{}_{:04}.npz'.format(self.mode, idx)) for idx in range(376) if os.path.exists(os.path.join(self.path, '{}_{:04}.npz'.format(self.mode, idx))) ]
        self.num_samples = np.rint(self.num_sample_points*self.sample_distribution ).astype(np.uint32)
        self.runs= runs

    def create_smpl(self, pose_in):
        pass

    def __len__(self):
        return self.runs


    def __getitem__(self, idx):


        seq_pose = np.load(self.path)['pose_body'].astype(np.float32)[:, :63]
        seq_pose = seq_pose.reshape(len(seq_pose), 21, 3)
        # change axis angle to quaternion
        quat_pose = axis_angle_to_quaternion_np(seq_pose)

        if self.mode == 'ref':
            return {'pose': quat_pose}

        # sample N poses from quat_pose, add noise and normalize
        samples_poses = []
        for i, num in enumerate(self.num_samples):
            indices = np.random.randint(0, len(quat_pose), num)
            sampled_pose = quat_pose[indices]

            #both methods are fine
            # sampled_pose = sampled_pose + self.sigma[i]*np.random.rand(21,4)*sampled_pose
            sampled_pose = sampled_pose + self.sigma[i]*np.random.rand(21,4)
            sampled_pose = sampled_pose / np.linalg.norm(sampled_pose, axis=2, keepdims=True)

            #move all points to single hemisphere
            sampled_pose, _ = quat_flip(sampled_pose)  
            samples_poses.extend(sampled_pose)
        sampled_poses = np.array(samples_poses)
        # sampled_poses = quat_doublecover(sampled_poses)
        assert len(sampled_poses) == self.num_sample_points

        return {'pose': sampled_poses, 'org_pose': quat_pose[indices]}

    def get_loader(self, shuffle=False):

        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle, drop_last=True)

    # def worker_init_fn(self, worker_id):
    #     random_data = os.urandom(4)
    #     base_seed = int.from_bytes(random_data, byteorder="big")
    #     np.random.seed(base_seed + worker_id)

class AmassData(Dataset):


    def __init__(self,  data_path,batch_size=100, num_workers=3, smpl=False, device='cuda:0'):

        self.path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device=device

        self.poses =np.load(self.path)['pose_body'].astype(np.float32)


    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):

        """read pose and distance data"""
        poses = self.poses[idx]
        # append zeros for hand keypoints
        poses_r = np.zeros(69).astype(np.float32)
        poses_r[:63] = poses
        # out_obj = self.bm( betas=self.betas,body_pose=poses)
    


        return {'pose': poses_r}

    def get_loader(self, shuffle =False):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

def sample_poses(root_dir, mode='train'):

    amass_datas = amass_splits[mode]
    tmp = 0
    all_pose = []
    for amass_data in amass_datas:
        ds_dir = os.path.join(root_dir,amass_data)
        seqs = sorted(os.listdir(ds_dir))


        for seq in seqs:
            if not 'npz' in seq:
                continue
            data = np.memmap(os.path.join(ds_dir, seq))
            tmp+= data.size
            print('cumulative....', tmp)
            data_val = np.load(os.path.join(ds_dir, seq))['pose_body']
            all_pose.extend(data_val.reshape(len(data_val), 21, 3))
    print('total size....', tmp)
    data_all = np.array(all_pose)
    index = faiss.index_factory(63, "Flat",faiss.METRIC_INNER_PRODUCT)
    index.train(data_all)
    index.add(data_all)
    xq = data_all[:10]
    distances, neighbors = index.search(xq.astype(np.float32), 5)

if __name__ == "__main__":


    posendf_data_dir = '/BS/humanpose/static00/data/PoseNDF_raw/smpl_h'

    sample_poses(posendf_data_dir, mode='train')