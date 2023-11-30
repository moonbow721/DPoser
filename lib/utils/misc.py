import torch
import numpy as np
import torch.nn.functional as F

from lib.body_model.utils import BodyPartIndices
from lib.body_model import constants
from lib.dataset.AMASS import N_POSES
from lib.utils.transforms import rot6d_to_axis_angle


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


def create_mask(body_poses, part='legs', observation_type='noise'):
    assert len(body_poses.shape) == 2 and body_poses.shape[1] % N_POSES == 0
    rot_N = body_poses.shape[1] // N_POSES
    assert rot_N in [3, 6]
    # for axis-angle or rot6d
    mask_joints = getattr(BodyPartIndices, part)
    mask = body_poses.new_ones(body_poses.shape)
    mask_indices = torch.tensor(mask_joints).view(-1, 1) * rot_N + torch.arange(rot_N).view(1, -1)
    mask_indices = mask_indices.flatten()
    mask[:, mask_indices] = 0

    # masked data as Gaussian noise
    observation = body_poses.clone()
    if observation_type == 'noise':
        observation[:, mask_indices] = torch.randn_like(observation[:, mask_indices])
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


def linear_interpolation(A, B, frames):
    alpha = torch.linspace(0, 1, frames, device=A.device)[:, None]
    interpolated = (1 - alpha) * A + alpha * B
    return interpolated


def slerp_interpolation(A, B, frames):
    omega = torch.acos((A * B).sum() / (torch.norm(A) * torch.norm(B)))
    alpha = torch.linspace(0, 1, frames, device=A.device)[:, None]
    slerped = (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * A + (
            torch.sin(alpha * omega) / torch.sin(omega)) * B
    return slerped


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