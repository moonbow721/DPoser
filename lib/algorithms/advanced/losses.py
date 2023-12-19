# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from lib.dataset.AMASS import N_POSES
from lib.utils.misc import linear_interpolation
from lib.utils.transforms import rot6d_to_axis_angle
from . import utils as mutils
from .sde_lib import VESDE, VPSDE


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        # default decay=0, lr=2e-4, beta1=0.9, same as my setting
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
        optimizer.step()

    return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=False, continuous=True, likelihood_weighting=False, eps=1e-5,
                    return_data=False, denoise_steps=5):
    """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.
    return_data: for outer auxiliary loss
  Returns:
    A loss function.
  """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition, mask):
        """Compute the loss function.

    Args:
      model: A score model.
      batch: [B, j*3] or [B, j*6]
      condition: None
      mask: None or same shape as condition
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
        def multi_step_denoise(x_t, t, t_end, N=10):
            time_traj = linear_interpolation(t, t_end, N + 1)
            x_current = x_t
            for i in range(N):
                t_current = time_traj[i]
                t_before = time_traj[i + 1]
                alpha_current, sigma_current = sde.return_alpha_sigma(t_current)
                alpha_before, sigma_before = sde.return_alpha_sigma(t_before)
                score = score_fn(x_current, t_current, condition=condition, mask=mask)
                if i == 0:
                    score_return = score
                score = -score * sigma_current[:, None]  # score to noise prediction
                x_current = alpha_before / alpha_current * (x_current - sigma_current[:, None] * score) + sigma_before[
                                                                                                          :,
                                                                                                          None] * score
            return score_return, x_current

        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        # prior t0 --> sde.T
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps  # [B]
        z = torch.randn_like(batch)  # [B, j*3]
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z  # [B, j*3]

        if return_data:
            # return estimated clean sample for auxiliary loss
            alpha, sigma = sde.return_alpha_sigma(t)
            SNR = alpha / sigma[:, None]
            score, estimated_data = multi_step_denoise(perturbed_data, t, t_end=t/(2*denoise_steps), N=denoise_steps)
        else:
            score = score_fn(perturbed_data, t, condition, mask)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None] + z)  # [B, j*3]
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)  # [B]
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = torch.mean(losses)
        if return_data:
            return loss, {'clean_sample': estimated_data, 'SNR': SNR, 't': t}
        else:
            return loss

    return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition, mask):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = torch.randn_like(batch) * sigmas[:, None]
        perturbed_data = noise + batch
        score = model_fn(perturbed_data, labels, condition, mask)
        target = -noise / (sigmas ** 2)[:, None]
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

    def loss_fn(model, batch, condition, mask):
        model_fn = mutils.get_model_fn(model, train=train)
        labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = torch.randn_like(batch)
        perturbed_data = sqrt_alphas_cumprod[labels, None] * batch + \
                         sqrt_1m_alphas_cumprod[labels, None] * noise
        score = model_fn(perturbed_data, labels, condition, mask)
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, continuous=True,
                likelihood_weighting=False, auxiliary_loss=False,
                denormalize=None, body_model=None, rot_rep='rot6d', denoise_steps=5):
    """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
    if continuous:
        loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean,
                                  continuous=True, likelihood_weighting=likelihood_weighting,
                                  return_data=auxiliary_loss, denoise_steps=denoise_steps)
    else:
        assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
        else:
            raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")
    if auxiliary_loss:
        assert denormalize is not None and body_model is not None

    l2_loss = nn.MSELoss(reduction='none')

    def step_fn(state, batch, condition=None, mask=None):
        """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            # basic diffusion loss
            if not auxiliary_loss:
                loss = loss_fn(model, batch, condition, mask)
                loss_dict = {'step_loss': loss, 'score_loss': loss}
            # auxiliary loss
            else:
                score_loss, data = loss_fn(model, batch, condition, mask)
                weight = torch.log(1.0 + data['SNR'])  # [b, 1]
                # weight = 1.0
                estimate = denormalize(data['clean_sample'])
                batch = denormalize(batch)
                if rot_rep == 'rot6d':
                    estimate = rot6d_to_axis_angle(estimate.reshape(-1, 6)).reshape(-1, N_POSES * 3)
                    batch = rot6d_to_axis_angle(batch.reshape(-1, 6)).reshape(-1, N_POSES * 3)
                # The bottleneck of training, it costs 10 times slower than multi-step denoising (even 10steps)
                gt_body = body_model(pose_body=batch)
                pred_body = body_model(pose_body=estimate)
                loss_v2v = torch.mean(weight * l2_loss(gt_body.v, pred_body.v).sum(dim=-1))     # sum along x-y-z
                loss_j2j = torch.mean(weight * l2_loss(gt_body.Jtr, pred_body.Jtr).sum(dim=-1))     # sum along x-y-z
                loss = score_loss + loss_v2v + loss_j2j
                loss_dict = {'step_loss': loss, 'score_loss': score_loss, 'v2v_loss': loss_v2v, 'j2j_loss': loss_j2j}

            loss.backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, condition, mask)
                ema.restore(model.parameters())
                loss_dict = {'step_loss': loss, 'score_loss': loss}

        return loss_dict

    return step_fn
