import numpy as np
import torch


from torch import nn
from torch.nn import functional as F

from lib.utils.transforms import mat3x3_to_axis_angle


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        """
        :param module_input: (N, 6)
        :return: (N, 3, 3)
        """
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class NormalDistDecoder(nn.Module):
    def __init__(self, num_feat_in, latentD):
        super(NormalDistDecoder, self).__init__()

        self.mu = nn.Linear(num_feat_in, latentD)
        self.logvar = nn.Linear(num_feat_in, latentD)

    def forward(self, Xout):
        return torch.distributions.normal.Normal(self.mu(Xout), F.softplus(self.logvar(Xout)))


class VPoser(nn.Module):
    def __init__(self, model_config, n_poses=21, pose_dim=3):
        super(VPoser, self).__init__()
        assert pose_dim == 3, 'Only axis representation are supported'
        num_neurons, self.latentD = model_config.num_neurons, model_config.latentD
        self.num_neurons = num_neurons
        self.num_joints = n_poses
        n_features = self.num_joints * pose_dim

        self.encoder_net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body):
        """
        pose_body: [B, j*3]
        Return: [B, latent]
        """
        pose_body = pose_body.view(pose_body.shape[0], -1)
        return self.encoder_net(pose_body)

    def decode(self, Zin):
        """
        Zin: [B, latent]
        Return: results dict including {pose_body [B, j, 3], pose_body_matrot [B, j, 9]}
        """
        bs = Zin.shape[0]
        prec = self.decoder_net(Zin)

        return {
            'pose_body': mat3x3_to_axis_angle(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }

    def forward(self, pose_body):
        """
        pose_body: [B, j*3]
        Return: results dict including {pose_body, pose_body_matrot,
                poZ_body_mean [], poZ_body_std, q_z}
        """

        q_z = self.encode(pose_body)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results

    def sample_poses(self, num_poses, seed=None):
        np.random.seed(seed)

        some_weight = next(self.parameters())
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.randn(num_poses, self.latentD, dtype=dtype, device=device)

        return self.decode(Zgen)


class CVPoser(nn.Module):
    def __init__(self, model_config, n_poses=21, pose_dim=3):
        super(CVPoser, self).__init__()
        assert pose_dim == 3, 'Only axis representation are supported'
        num_neurons, self.latentD = model_config.num_neurons, model_config.latentD
        self.num_neurons = num_neurons
        self.num_joints = n_poses
        n_features = self.num_joints * pose_dim
        # Encoder input（pose + condition）
        expanded_input_size = n_features * 2  # 因为pose_body和condition被拼接

        self.encoder_net = nn.Sequential(
            nn.BatchNorm1d(expanded_input_size),
            nn.Linear(expanded_input_size, num_neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_neurons),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.Linear(num_neurons, num_neurons),
            NormalDistDecoder(num_neurons, self.latentD)
        )

        # New decoder
        expanded_decoder_input_size = self.latentD + n_features
        self.decoder_net = nn.Sequential(
            nn.Linear(expanded_decoder_input_size, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def encode(self, pose_body, condition):
        """
        pose_body: [B, j*3]
        condition: [B, j*3] (partial poses)
        Return: [B, latent]
        """
        combined_input = torch.cat([pose_body, condition], dim=1)
        combined_input = combined_input.view(combined_input.shape[0], -1)
        return self.encoder_net(combined_input)

    def decode(self, Zin, condition):
        """
        Zin: [B, latent]
        condition: [B, j*3] (partial poses)
        Return: results dict including {pose_body [B, j, 3], pose_body_matrot [B, j, 9]}
        """
        bs = Zin.shape[0]
        combined_input = torch.cat([Zin, condition.view(Zin.shape[0], -1)], dim=1)
        prec = self.decoder_net(combined_input)

        return {
            'pose_body': mat3x3_to_axis_angle(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }

    def forward(self, pose_body, condition):
        """
        pose_body: [B, j*3]
        condition: [B, j*3]
        Return: results dict including {pose_body, pose_body_matrot,
                poZ_body_mean [], poZ_body_std, q_z}
        """
        q_z = self.encode(pose_body, condition)
        q_z_sample = q_z.rsample()
        decode_results = self.decode(q_z_sample, condition)
        decode_results.update({'poZ_body_mean': q_z.mean, 'poZ_body_std': q_z.scale, 'q_z': q_z})
        return decode_results

    # sample_poses now include conditional input
    def sample_poses(self, num_poses, condition, seed=None):
        """
        condition: [B, j*3]
        return [num_poses, B, j*3]
        """
        if seed is not None:
            np.random.seed(seed)
        bs = condition.shape[0]
        some_weight = next(self.parameters())
        dtype = some_weight.dtype
        device = some_weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.randn(num_poses*bs, self.latentD, dtype=dtype, device=device)
            if condition.ndim == 2:
                condition = condition.unsqueeze(0).repeat(num_poses, 1, 1)
            condition = condition.to(dtype=dtype, device=device)

        results = self.decode(Zgen, condition.view(bs*num_poses, -1))['pose_body'].reshape(num_poses*bs, -1)

        return results.view(num_poses, bs, -1)


# Just change network structure to avoid mode collapse
class CVPoser2(CVPoser):
    def __init__(self, model_config, n_poses=21, pose_dim=3):
        super(CVPoser2, self).__init__(model_config, n_poses, pose_dim)

        num_neurons, self.latentD = model_config.num_neurons, model_config.latentD
        self.decoder_net = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
            nn.Linear(num_neurons, self.num_joints * 6),
            ContinousRotReprDecoder(),
        )

    def decode(self, Zin, condition):
        """
        Zin: [B, latent]
        condition: [B, j*3] (partial poses)
        Return: results dict including {pose_body [B, j, 3], pose_body_matrot [B, j, 9]}
        """
        bs = Zin.shape[0]
        prec = self.decoder_net(Zin)

        return {
            'pose_body': mat3x3_to_axis_angle(prec.view(-1, 3, 3)).view(bs, -1, 3),
            'pose_body_matrot': prec.view(bs, -1, 9)
        }