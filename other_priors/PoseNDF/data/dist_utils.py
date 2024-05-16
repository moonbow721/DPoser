"""Quaternion euclidean, quaternion geodesic and v2v based distances"""

import smplx
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle,axis_angle_to_matrix
import ipdb

class euc():

    def __init__(self, batch_size,device='cuda',weighted=False):
        self.device= device

        self.batch_size = batch_size
        self.weighted = weighted
        joint_rank = torch.from_numpy(np.array([7,7,7,6,6,6,5,5,5,4,4,4,4,4,3,3,3,2,2,1,1])).float()
        self.joint_weights = torch.nn.functional.normalize(joint_rank,dim=0).to(device=self.device)

    def dist_calc(self, noise_quats, valid_quat,k_faiss, k_dist):
        noise_quats = noise_quats.unsqueeze(1).repeat(1, k_faiss, 1,1)

        dist = noise_quats -  valid_quat

        if self.weighted:
            geo_dis = torch.sum(self.joint_weights * torch.sqrt(torch.sum(dist * dist, dim=3)), dim=2)
        else:
            geo_dis = torch.mean(torch.sqrt(torch.sum(dist * dist, dim=3)), dim=2)
        geo_val, geo_idx = torch.topk(geo_dis, k=5, largest=False)

        return geo_val, geo_idx

class geo():

    def __init__(self, batch_size,device='cuda',weighted=False):
        self.device= device

        self.batch_size = batch_size
        self.weighted = weighted
        joint_rank = torch.from_numpy(np.array([7,7,7,6,6,6,5,5,5,4,4,4,4,4,3,3,3,2,2,1,1])).float()
        self.joint_weights = torch.nn.functional.normalize(joint_rank,dim=0).to(device=self.device)

    def dist_calc(self, noise_quats, valid_quat,k_faiss, k_dist):
        noise_quats = noise_quats.unsqueeze(1).repeat(1, k_faiss, 1,1)
        if self.weighted:
            geo_dis =torch.sum(self.joint_weights *(1 - torch.abs(torch.sum(valid_quat*noise_quats,dim=3))),dim=2) 
        else:
            geo_dis = torch.mean(1 - torch.abs(torch.sum(valid_quat*noise_quats,dim=3)),dim=2)    #https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
        geo_val, geo_idx = torch.topk(geo_dis, k=5, largest=False)

        return geo_val, geo_idx
