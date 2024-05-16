# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import pickle

import numpy as np

import torch
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_quaternion

DEFAULT_DTYPE = torch.float32


def create_prior(prior_type, **kwargs):
    if prior_type == 'gmm':
        prior = MaxMixturePrior(**kwargs)
    elif prior_type == 'l2':
        return L2Prior(**kwargs)
    elif prior_type == 'angle':
        return SMPLifyAnglePrior(**kwargs)
    elif prior_type == 'none' or prior_type is None:
        # Don't use any pose prior
        def no_prior(*args, **kwargs):
            return 0.0

        prior = no_prior
    else:
        raise ValueError('Prior {}'.format(prior_type) + ' is not implemented')
    return prior


class SMPLifyAnglePrior(nn.Module):
    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()

        # Indices for the roration angle of
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)

        angle_prior_signs = np.array([1, -1, -1, -1],
                                     dtype=np.float32 if dtype == torch.float32
                                     else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs,
                                         dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        ''' Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        '''
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] *
                         self.angle_prior_signs).pow(2)


class L2Prior(nn.Module):
    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        # l2_norm = torch.norm(module_input, p=2, dim=[-1], keepdim=True)
        return torch.mean(module_input.pow(2).sum(dim=-1)) * 0.1


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior',
                 num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16,
                 use_merged=True,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi) ** (69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        zero_hands = torch.zeros([pose.shape[0], 6], device=pose.device)
        pose_all = torch.cat([pose, zero_hands], dim=1)
        diff_from_mean = pose_all.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
                             torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)

        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas, *args):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)

    def sample(self, sample_num):
        """
        Sample from the Gaussian Mixture Model
        Args:
            sample_num: The number of samples
        Returns:
            samples: tensor with shape [sample_num, ...]
        """

        # Step 1: Choose a Gaussian component for each sample
        component_indices = np.random.choice(self.num_gaussians, size=sample_num,
                                             p=self.weights.squeeze().cpu().numpy())

        # Step 2: Sample from the chosen Gaussian distribution
        samples = []
        for i in component_indices:
            mean = self.means[i]
            cov = self.covs[i]
            sample = np.random.multivariate_normal(mean.cpu().numpy(), cov.cpu().numpy())
            samples.append(sample)

        samples = np.stack(samples)
        samples = torch.tensor(samples, dtype=self.means.dtype, device=self.means.device)

        return samples


def quat_flip(pose_in):
    is_neg = pose_in[:, :, 0] < 0
    pose_in[is_neg] = (-1) * pose_in[is_neg]
    return pose_in, is_neg


class Posendf(nn.Module):
    def __init__(self, config, ckpt_path):
        super().__init__()
        sys.path.insert(0, './other_priors/PoseNDF')
        from model.posendf import PoseNDF
        from configs.config import load_config
        opt = load_config(config)
        net = PoseNDF(opt)
        ckpt = torch.load(ckpt_path, map_location='cpu')['model_state_dict']
        net.load_state_dict(ckpt)
        net.eval()
        self.pose_prior = net

    def forward(self, pose, betas, *args):
        pose_quat = axis_angle_to_quaternion(pose.view(-1, 21, 3))
        pose_quat, _ = quat_flip(pose_quat)
        pose_quat = torch.nn.functional.normalize(pose_quat, dim=-1)

        dis_val = self.pose_prior(pose_quat, train=False)['dist_pred'] ** 2
        return torch.mean(dis_val) * 1e6


class VPoser(nn.Module):
    def __init__(self, support_dir):
        super().__init__()
        sys.path.insert(0, './other_priors/human_body_prior/src')
        from human_body_prior.tools.model_loader import load_vposer

        expr_dir = os.path.join(support_dir, 'vposer_v1_0')
        vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
        self.vposer = vposer_pt

    def forward(self, pose, betas, *args):
        assert len(pose.shape) == 2
        coding = self.vposer.encode(pose).mean
        l2_norm = torch.norm(coding, p=2, dim=[1], keepdim=True)
        return torch.mean(l2_norm) * 0.5

