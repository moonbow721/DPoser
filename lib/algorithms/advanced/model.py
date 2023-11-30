import functools

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

    return sigmas


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def get_act(config):
    """Get activation functions from the config file."""

    if config.model.nonlinearity.lower() == 'elu':
        return nn.ELU()
    elif config.model.nonlinearity.lower() == 'relu':
        return nn.ReLU()
    elif config.model.nonlinearity.lower() == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif config.model.nonlinearity.lower() == 'swish':
        return nn.SiLU()
    else:
        raise NotImplementedError('activation function does not exist!')


class TimeMLPs(torch.nn.Module):
    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64, n_blocks=2):
        super().__init__()
        dim = n_poses * pose_dim
        self.act = get_act(config)

        layers = [torch.nn.Linear(dim + 1, hidden_dim),
                  self.act]

        for _ in range(n_blocks):
            layers.extend([
                torch.nn.Linear(hidden_dim, hidden_dim),
                self.act,
                torch.nn.Dropout(p=config.model.dropout)
            ])

        layers.append(torch.nn.Linear(hidden_dim, dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t, condition=None, mask=None):
        return self.net(torch.cat([x, t[:, None]], dim=1))


class ScoreModelFC(nn.Module):
    """
    Independent condition feature projection layers for each block
    """

    def __init__(self, config, n_poses=21, pose_dim=6, hidden_dim=64,
                 embed_dim=32, n_blocks=2):
        super(ScoreModelFC, self).__init__()

        self.config = config
        self.n_poses = n_poses
        self.joint_dim = pose_dim
        self.n_blocks = n_blocks

        self.act = get_act(config)

        self.pre_dense = nn.Linear(n_poses * pose_dim, hidden_dim)
        self.pre_dense_t = nn.Linear(embed_dim, hidden_dim)
        self.pre_dense_cond = nn.Linear(hidden_dim, hidden_dim)
        self.pre_gnorm = nn.GroupNorm(32, num_channels=hidden_dim)
        self.dropout = nn.Dropout(p=config.model.dropout)

        # time embedding
        self.time_embedding_type = config.model.embedding_type.lower()
        if self.time_embedding_type == 'fourier':
            self.gauss_proj = GaussianFourierProjection(embed_dim=embed_dim, scale=config.model.fourier_scale)
        elif self.time_embedding_type == 'positional':
            self.posit_proj = functools.partial(get_timestep_embedding, embedding_dim=embed_dim)
        else:
            assert 0

        self.shared_time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
        )
        self.register_buffer('sigmas', torch.tensor(get_sigmas(config), dtype=torch.float))

        for idx in range(n_blocks):
            setattr(self, f'b{idx + 1}_dense1', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense1_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm1', nn.GroupNorm(32, num_channels=hidden_dim))

            setattr(self, f'b{idx + 1}_dense2', nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_dense2_t', nn.Linear(embed_dim, hidden_dim))
            setattr(self, f'b{idx + 1}_gnorm2', nn.GroupNorm(32, num_channels=hidden_dim))

        self.post_dense = nn.Linear(hidden_dim, n_poses * pose_dim)

    def forward(self, batch, t, condition=None, mask=None):
        """
        batch: [B, j*3] or [B, j*6]
        t: [B]
        Return: [B, j*3] or [B, j*6] same dim as batch
        """
        bs = batch.shape[0]

        # batch = batch.view(bs, -1)  # [B, j*3]

        # time embedding
        if self.time_embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = t
            temb = self.gauss_proj(torch.log(used_sigmas))
        elif self.time_embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = t
            used_sigmas = self.sigmas[t.long()]
            temb = self.posit_proj(timesteps)
        else:
            raise ValueError(f'time embedding type {self.time_embedding_type} unknown.')

        temb = self.shared_time_embed(temb)

        h = self.pre_dense(batch)
        h += self.pre_dense_t(temb)
        h = self.pre_gnorm(h)
        h = self.act(h)
        h = self.dropout(h)

        for idx in range(self.n_blocks):
            h1 = getattr(self, f'b{idx + 1}_dense1')(h)
            h1 += getattr(self, f'b{idx + 1}_dense1_t')(temb)
            h1 = getattr(self, f'b{idx + 1}_gnorm1')(h1)
            h1 = self.act(h1)
            # dropout, maybe
            h1 = self.dropout(h1)

            h2 = getattr(self, f'b{idx + 1}_dense2')(h1)
            h2 += getattr(self, f'b{idx + 1}_dense2_t')(temb)
            h2 = getattr(self, f'b{idx + 1}_gnorm2')(h2)
            h2 = self.act(h2)
            # dropout, maybe
            h2 = self.dropout(h2)

            h = h + h2

        res = self.post_dense(h)  # [B, j*3]

        ''' normalize the output '''
        if self.config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((bs, 1))
            res = res / used_sigmas

        return res
