import math
import torch
from torch import nn
from tqdm import tqdm

from lib.algorithms.advanced import sde_lib, sampling
from lib.algorithms.advanced import utils as mutils
from lib.algorithms.advanced.model import ScoreModelFC
from lib.algorithms.ema import ExponentialMovingAverage
from lib.body_model import constants
from lib.body_model.fitting_losses import camera_fitting_loss, body_fitting_loss
from lib.dataset.AMASS import N_POSES, Posenormalizer
from lib.utils.generic import import_configs
from lib.utils.misc import linear_interpolation


class DPoser(nn.Module):
    def __init__(self, batch_size=32, config_path='', args=None):
        super().__init__()
        self.device = args.device
        self.batch_size = batch_size
        config = import_configs(config_path)

        self.Normalizer = Posenormalizer(
            data_path=f'{args.dataset_folder}/{args.version}/train',
            min_max=config.data.min_max, rot_rep=config.data.rot_rep, device=args.device)

        diffusion_model = self.load_model(config, args)
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                N=config.model.num_scales)
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        sde.N = args.sde_N  # fewer sampling steps

        self.sde = sde
        self.score_fn = mutils.get_score_fn(sde, diffusion_model, train=False, continuous=config.training.continuous)
        self.rsde = sde.reverse(self.score_fn, False)
        # L2 loss
        self.loss_fn = nn.MSELoss(reduction='none')
        self.timesteps = torch.linspace(self.sde.T, 1e-3, self.sde.N, device=self.device)

    def load_model(self, config, args):
        POSE_DIM = 3 if config.data.rot_rep == 'axis' else 6
        model = ScoreModelFC(
            config,
            n_poses=N_POSES,
            pose_dim=POSE_DIM,
            hidden_dim=config.model.HIDDEN_DIM,
            embed_dim=config.model.EMBED_DIM,
            n_blocks=config.model.N_BLOCKS,
        )
        model.to(self.device)
        model.eval()
        map_location = {'cuda:0': self.device}
        checkpoint = torch.load(args.ckpt_path, map_location=map_location)
        ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema'])
        return model

    def one_step_denoise(self, x_t, t):
        drift, diffusion, alpha, sigma_2, score = self.rsde.sde(x_t, t, guide=True)
        x_0_hat = (x_t + sigma_2[:, None] * score) / alpha
        SNR = alpha / torch.sqrt(sigma_2)[:, None]

        return x_0_hat.detach(), SNR

    def multi_step_denoise(self, x_t, t, t_end, N=10):
        time_traj = linear_interpolation(t, t_end, N + 1)
        x_current = x_t

        for i in range(N):
            t_current = time_traj[i]
            t_before = time_traj[i + 1]
            alpha_current, sigma_current = self.sde.return_alpha_sigma(t_current)
            alpha_before, sigma_before = self.sde.return_alpha_sigma(t_before)
            score = self.score_fn(x_current, t_current, condition=None, mask=None)
            score = -score * sigma_current[:, None]  # score to noise prediction
            x_current = alpha_before / alpha_current * (x_current - sigma_current[:, None] * score) + sigma_before[
                                                                                                      :,
                                                                                                      None] * score
        alpha, sigma = self.sde.return_alpha_sigma(time_traj[0])
        SNR = alpha / sigma[:, None]
        return x_current.detach(), SNR

    def DPoser_loss(self, x_0, vec_t, multi_denoise=False):
        # x_0: [B, j*6], vec_t: [B], quan_t: [1]
        z = torch.randn_like(x_0)
        mean, std = self.sde.marginal_prob(x_0, vec_t)
        perturbed_data = mean + std[:, None] * z  #
        if multi_denoise:
            denoise_data, SNR = self.multi_step_denoise(perturbed_data, vec_t, t_end=vec_t / (2 * 5), N=5)
        else:
            denoise_data, SNR = self.one_step_denoise(perturbed_data, vec_t)
        weight = 0.5 * torch.sqrt(1+SNR)
        # weight = 0.5
        loss = torch.sum(weight * self.loss_fn(x_0, denoise_data)) / self.batch_size

        return loss

    def forward(self, poses, betas, quan_t):
        poses = self.Normalizer.offline_normalize(poses[:, :N_POSES * 3], from_axis=True)

        t = self.timesteps[quan_t]
        vec_t = torch.ones(self.batch_size, device=self.device) * t
        prior_loss = self.DPoser_loss(poses, vec_t)
        return prior_loss


class SMPLify:
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 body_model,
                 step_size=1e-2,
                 batch_size=32,
                 num_iters=100,
                 focal_length=5000,
                 args=None):
        self.smpl = body_model
        # Store options
        self.device = args.device
        self.focal_length = focal_length
        self.step_size = step_size

        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters

        self.pose_prior = DPoser(batch_size, args.config_path, args)
        self.sde_N = args.sde_N
        # TODO: try different time strategy parameters
        self.time_strategy = args.time_strategy
        self.sample_time = round(args.sde_N * 0.9)
        self.sample_trun = 20.0

        # TODO: set different loss weights
        self.loss_weights = {'pose_prior_weight': [50, 20, 10, 5, 2],
                             'shape_prior_weight': [50, 20, 10, 5, 2],
                             'angle_prior_weight': [150, 50, 30, 15, 5],
                             }
        self.stages = len(self.loss_weights['pose_prior_weight'])

    def sample_discrete_time(self, iteration):
        total_steps = self.stages * self.num_iters
        if self.time_strategy == '1':  # not recommend
            quan_t = torch.randint(self.sde_N, [1])
        elif self.time_strategy == '2':
            quan_t = torch.tensor(self.sample_time)
        elif self.time_strategy == '3':
            quan_t = self.sde_N - math.floor(
                torch.tensor(total_steps - iteration - 1) * (
                        self.sde_N / (self.sample_trun * total_steps))) - 5
        else:
            raise NotImplementedError

        return quan_t

    def __call__(self, init_pose, init_betas, init_cam_t, camera_center, keypoints_2d):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            camera_center: Camera center location
            keypoints_2d: Keypoints used for the optimization
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
            reprojection_loss: Final joint reprojection loss
        """
        batch_size = init_pose.shape[0]

        # Make camera translation a learnable parameter
        camera_translation = init_cam_t.clone()

        # Get joint confidence
        joints_2d = keypoints_2d[:, :, :2]
        joints_conf = keypoints_2d[:, :, -1]

        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Step 1: Optimize camera translation and body orientation
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    pose2rot=True,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length)
            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        # Fix camera translation after optimizing camera
        camera_translation.requires_grad = False

        # Step 2: Optimize body joints
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        body_opt_params = [body_pose, betas, global_orient]

        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))

        stage_weights = [dict(zip(self.loss_weights.keys(), vals)) for vals in zip(*self.loss_weights.values())]

        for stage, current_weights in enumerate(tqdm(stage_weights, desc='Stage')):
            for i in range(self.num_iters):
                smpl_output = self.smpl(betas=betas,
                                        body_pose=body_pose,
                                        global_orient=global_orient,
                                        pose2rot=True,
                                        transl=camera_translation)

                model_joints = smpl_output.joints
                quan_t = self.sample_discrete_time(iteration=stage * self.num_iters + i)

                loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior, quan_t=quan_t,
                                         focal_length=self.focal_length,
                                         **current_weights,
                                         verbose=False)

                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss value
        with torch.no_grad():

            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    pose2rot=True,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            quan_t = self.sample_discrete_time(iteration=self.num_iters - 1)
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior, quan_t=quan_t,
                                                  focal_length=self.focal_length,
                                                  output='reprojection', verbose=False)

        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return pose, betas, camera_translation, reprojection_loss
