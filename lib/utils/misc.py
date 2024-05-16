import os
import random

import torch
import torch.nn.functional as F
import numpy as np

from lib.body_model.utils import BodyPartIndices, HandPartIndices
from lib.body_model import constants
from lib.utils.transforms import rot6d_to_axis_angle


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    elif isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    return data.to(device)


def add_noise(gts, std=0.5, noise_type='gaussian'):
    if std == 0.0:
        return gts

    if noise_type == 'gaussian':
        noise = std * torch.randn(*gts.shape, device=gts.device)
        gts = gts + noise
    elif noise_type == 'uniform':
        # a range of [-0.5std, 0.5std]
        noise = std * (torch.rand(*gts.shape, device=gts.device) - 0.5)
        gts = gts + noise
    else:
        raise NotImplementedError
    return gts


def create_mask(body_poses, part='legs', model='body', observation_type='noise'):
    # body_poses: [batchsize, 3*N_POSES] (axis-angle) or [batchsize, 6*N_POSES] (rot6d)
    if model == 'body':
        N_POSES = 21
        PartIndices = BodyPartIndices
    elif model == 'hand':
        N_POSES = 15
        PartIndices = HandPartIndices
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_poses.shape) == 2 and body_poses.shape[1] % N_POSES == 0
    rot_N = body_poses.shape[1] // N_POSES
    assert rot_N in [3, 6]

    mask_joints = PartIndices.get_indices(part)
    mask = body_poses.new_ones(body_poses.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_joints, dtype=torch.long).view(-1, 1) * rot_N + torch.arange(rot_N).view(1, -1)
    mask_indices = mask_indices.flatten()
    mask[:, mask_indices] = 0

    # masked data as Gaussian noise
    observation = body_poses.clone()
    if observation_type == 'noise':
        observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices])
    elif observation_type == 'zeros':
        observation[:, mask_indices] = torch.zeros_like(observation[:, mask_indices])
    # load the mean pose as observation
    else:
        batch_size = body_poses.shape[0]
        smpl_mean_params = np.load(constants.SMPL_MEAN_PATH)
        rot6d_body_poses = torch.tensor(smpl_mean_params['pose'][6:,], dtype=torch.float32, device=body_poses.device)  # [138]
        axis_body_pose = rot6d_to_axis_angle(rot6d_body_poses.reshape(-1, 6)).reshape(-1)   # [69]
        if rot_N == 3:
            observation[:, mask_indices] = axis_body_pose[None, mask_indices].repeat(batch_size, 1)
        elif rot_N == 6:
            observation[:, mask_indices] = rot6d_body_poses[None, mask_indices].repeat(batch_size, 1)
        else:
            raise NotImplementedError

    return mask, observation


def create_random_mask(body_poses, min_mask_rate=0.2, max_mask_rate=0.4, model='body', observation_type='noise'):
    # body_poses: [batchsize, 3*N_POSES] (axis-angle) or [batchsize, 6*N_POSES] (rot6d)
    if model == 'body':
        N_POSES = 21
    elif model == 'hand':
        N_POSES = 15
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_poses.shape) == 2 and body_poses.shape[1] % N_POSES == 0
    rot_N = body_poses.shape[1] // N_POSES
    assert rot_N in [3, 6]

    mask_rate = random.uniform(min_mask_rate, max_mask_rate)
    num_joints_to_mask = int(round(mask_rate * N_POSES))
    if num_joints_to_mask == 0:
        return body_poses.new_ones(body_poses.shape), body_poses
    mask_joints = random.sample(range(N_POSES), num_joints_to_mask)
    mask = body_poses.new_ones(body_poses.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_joints).view(-1, 1) * rot_N + torch.arange(rot_N).view(1, -1)
    mask_indices = mask_indices.flatten()
    mask[:, mask_indices] = 0

    # masked data as Gaussian noise
    observation = body_poses.clone()
    if observation_type == 'noise':
        observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices])
    else:
        observation[:, mask_indices] = torch.zeros_like(observation[:, mask_indices])

    return mask, observation


def create_joint_mask(body_joints, part='legs', model='body'):
    # body_joints: [batchsize, 22, 3]
    if model == 'body':
        N_POSES = 21
        PartIndices = BodyPartIndices
    elif model == 'hand':
        N_POSES = 15
        PartIndices = HandPartIndices
    else:
        raise ValueError(f'Unknown model: {model}')
    assert len(body_joints.shape) == 3 and body_joints.shape[1] == N_POSES + 1  # +1 for root joint

    mask_indices = [x+1 for x in PartIndices.get_indices(part)]
    mask = body_joints.new_ones(body_joints.shape, dtype=torch.bool)
    mask_indices = torch.tensor(mask_indices)
    mask[:, mask_indices, :] = False

    # masked data as Gaussian noise
    observation = body_joints.clone()
    observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices]) * 0.1

    return mask, observation


def create_stable_mask(mask, eps=1e-6):
    stable_mask = torch.where(mask,
                              torch.tensor(1.0, dtype=torch.float32, device=mask.device),
                              torch.tensor(eps, dtype=torch.float32, device=mask.device))

    return stable_mask


def lerp(A, B, steps):
    A, B = A.unsqueeze(0), B.unsqueeze(0)
    alpha = torch.linspace(0, 1, steps, device=A.device)
    while alpha.dim() < A.dim():
        alpha = alpha.unsqueeze(-1)

    interpolated = (1 - alpha) * A + alpha * B
    return interpolated


def slerp(v1, v2, steps, DOT_THR=0.9995, zdim=-1):
    """
    SLERP for pytorch tensors interpolating `v1` to `v2` over `num_frames`.

    :param v1: Start vector.
    :param v2: End vector.
    :param num_frames: Number of frames in the interpolation.
    :param DOT_THR: Threshold for parallel vectors.
    :param zdim: Dimension over which to compute norms and find angles.
    :return: Interpolated frames.
    """
    # Normalize the input vectors
    v1_norm = v1 / torch.norm(v1, dim=zdim, keepdim=True)
    v2_norm = v2 / torch.norm(v2, dim=zdim, keepdim=True)

    # Dot product
    dot = (v1_norm * v2_norm).sum(zdim)

    # Mask for vectors that are too close to parallel
    parallel_mask = torch.abs(dot) > DOT_THR

    # SLERP interpolation
    theta = torch.acos(dot).unsqueeze(0)
    alpha = torch.linspace(0, 1, steps, device=v1.device)
    while alpha.dim() < theta.dim():
        alpha = alpha.unsqueeze(-1)
    theta_t = theta * alpha
    sin_theta = torch.sin(theta)
    sin_theta_t = torch.sin(theta_t)

    s1 = torch.sin(theta - theta_t) / sin_theta
    s2 = sin_theta_t / sin_theta
    slerp_res = (s1.unsqueeze(zdim) * v1) + (s2.unsqueeze(zdim) * v2)

    # LERP interpolation
    lerp_res = lerp(v1, v2, steps)

    # Combine results based on the parallel mask
    combined_res = torch.where(parallel_mask.unsqueeze(zdim), lerp_res, slerp_res)

    return combined_res


def moving_average(data, window_size):
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(data.device)

    data = data.transpose(0, 1).unsqueeze(1)

    smoothed_data = F.conv1d(data, kernel, padding=window_size//2)

    smoothed_data = smoothed_data.squeeze(1).transpose(0, 1)
    return smoothed_data


def gaussian_smoothing(data, window_size, sigma):
    kernel = torch.arange(window_size).float() - window_size // 2
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= kernel.sum()

    kernel = kernel.unsqueeze(0).unsqueeze(0).to(data.device)
    data = data.transpose(0, 1).unsqueeze(1)

    smoothed_data = F.conv1d(data, kernel, padding=window_size//2)

    smoothed_data = smoothed_data.squeeze(1).transpose(0, 1)
    return smoothed_data


def rbf_kernel(X, Y, gamma=-1, ad=1):
    # X and Y should be tensors with shape (batch_size, data_dim)
    # gamma is a hyperparameter controlling the width of the RBF kernel

    # Compute the pairwise squared Euclidean distances between the samples
    with torch.cuda.amp.autocast():
        dists = torch.cdist(X, Y, p=2) ** 2

    if gamma < 0:  # use median trick
        gamma = torch.median(dists)
        gamma = torch.sqrt(0.5 * gamma / np.log(X.size(0) + 1))
        gamma = 1 / (2 * gamma ** 2)
    else:
        gamma = gamma * ad

    # Compute the RBF kernel using the squared distances and gamma
    K = torch.exp(-gamma * dists)

    # Compute the gradient of the RBF kernel with respect to X
    dK = -2 * gamma * K.unsqueeze(-1) * (X.unsqueeze(1) - Y.unsqueeze(0))
    dK_dX = torch.sum(dK, dim=1)

    return K, dK_dX


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Generate two random latent vectors
    A = torch.randn(5, 5, 3)
    B = torch.randn(5, 5, 3)
    # Number of frames for interpolation
    num_frames = 100

    # Perform interpolations
    linear_results = lerp(A, B, num_frames)[:, 2, 2]
    slerp_results = slerp(A, B, num_frames)[:, 2, 2]
    print(linear_results.shape, slerp_results.shape)

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Linear Interpolation Plot
    plt.subplot(1, 2, 1)
    for i in range(3):  # Loop over each dimension
        plt.plot(linear_results[:, i].numpy(), label=f'Dim {i + 1}')
    plt.title("Linear Interpolation")
    plt.legend()

    # SLERP Interpolation Plot
    plt.subplot(1, 2, 2)
    for i in range(3):  # Loop over each dimension
        plt.plot(slerp_results[:, i].numpy(), label=f'Dim {i + 1}')
    plt.title("SLERP Interpolation")
    plt.legend()

    plt.savefig('./interpolation.png')