import pickle
import sys
import torch

from gan_lib.models.gan import Generator, Discriminator
from gan_lib.losses.vanilla_gan import DiscrLoss
from gan_lib.functional.sampling import generate_random_input
from torch import nn

sys.path.insert(0, '/data3/ljz24/projects/3d/DPoser')
from lib.body_model import constants
from lib.body_model.fitting_losses import camera_fitting_loss, body_fitting_loss

hidden_poseall = 64
poseall_num_layers = 2
latent_size = 32
num_joints = 21


class DiscriminatorPrior(nn.Module):
    def __init__(self, net):
        super(DiscriminatorPrior, self).__init__()
        self.net = net
        self.discr_loss = DiscrLoss()

    def forward(self, pose, betas, *args):
        discr_outs = self.net(pose, input_type='aa')
        return self.discr_loss(discr_outs, 1)


class SphericalPrior(nn.Module):
    def __init__(self, ):
        super(SphericalPrior, self).__init__()

    def forward(self, latents):
        norms = latents.norm(p=2, dim=1, keepdim=True)
        loss = torch.mean((norms - 1) ** 2)
        return loss


class SMPLify:
    """Implementation of single-stage SMPLify."""

    def __init__(self,
                 body_model,
                 step_size=1e-2,
                 batch_size=32,
                 num_iters=100,
                 focal_length=5000,
                 side_view_thsh=25.0,
                 args=None,
                 init_latent=None,):
        self.smpl = body_model
        # Store options
        self.device = args.device
        self.focal_length = focal_length
        self.side_view_thsh = side_view_thsh
        self.step_size = step_size
        self.latents = init_latent if init_latent is not None else (
            generate_random_input(batch_size, latent_size, latent_space_type='S', device=self.device))
        # Ignore the the following joints for the fitting process
        ign_joints = ['OP Neck', 'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip']
        self.ign_joints = [constants.JOINT_IDS[i] for i in ign_joints]
        self.num_iters = num_iters
        self.prior_name = args.prior

        ckpt = './output/ganS/gan_S_amass/ckpt_0300.pth'  # path to pretrained generative model
        if args.prior == 'generator':
            net = Generator(latentD=latent_size, num_joints=num_joints)
            ckpt = torch.load(ckpt, map_location='cpu')['generator_state_dict']
            net.load_state_dict(ckpt)
            net.eval()
            self.generator = net.to(self.device)
            # fix generator
            for param in self.generator.parameters():
                param.requires_grad = False
            self.latent_prior = SphericalPrior()
            self.pose_prior = None
        elif args.prior == 'discriminator':
            net = Discriminator(num_joints=num_joints,
                                hidden_poseall=hidden_poseall,
                                poseall_num_layers=poseall_num_layers,)
            ckpt = torch.load(ckpt, map_location='cpu')['discriminator_state_dict']
            net.load_state_dict(ckpt)
            net.eval()
            self.discriminator = net.to(self.device)
            self.pose_prior = DiscriminatorPrior(self.discriminator)
        else:
            self.pose_prior = None

        self.loss_weights = {'pose_prior_weight': [50, 20, 10, 5, 2],
                             'shape_prior_weight': [50, 20, 10, 5, 2],
                             'angle_prior_weight': [150, 50, 30, 15, 5],
                             'coll_loss_weight': [0, 0, 0, 0.01, 1.0],
                             }
        self.stages = len(self.loss_weights['pose_prior_weight'])
        self.interpenetration = args.interpenetration
        self.search_tree, self.pen_distance, self.filter_faces = None, None, None
        self.body_model_faces = self.smpl.bm.faces_tensor.view(-1).to(self.device)

    def sample_continuous_time(self, iteration):

        return 0

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
        joints_2d = keypoints_2d[:, :, :2].clone()
        joints_conf = keypoints_2d[:, :, -1].clone()

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
        camera_optimizer = torch.optim.Adam(camera_opt_params, lr=1e-2, betas=(0.9, 0.999))

        for i in range(self.num_iters):
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    pose2rot=True,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            loss = camera_fitting_loss(model_joints, camera_translation,
                                       init_cam_t, camera_center,
                                       joints_2d, joints_conf, focal_length=self.focal_length, part='body')

            camera_optimizer.zero_grad()
            loss.backward()
            camera_optimizer.step()

        left_shoulder_idx, right_shoulder_idx = 2, 5
        shoulder_dist = torch.dist(joints_2d[:, left_shoulder_idx],
                                   joints_2d[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < self.side_view_thsh

        # Step 2: Optimize body joints
        # For joints ignored during fitting, set the confidence to 0
        joints_conf[:, self.ign_joints] = 0.

        inputs = (global_orient, body_pose, betas, camera_translation, camera_center, joints_2d, joints_conf)
        if try_both_orient:
            pose, betas, camera_translation, reprojection_loss = self.optimize_and_compare(*inputs)
        else:
            reprojection_loss, (pose, betas, camera_translation, _) = self.optimize_body(*inputs)

        return pose, betas, camera_translation, reprojection_loss

    def optimize_and_compare(self, global_orient, body_pose, betas, camera_translation, camera_center, joints_2d,
                             joints_conf):
        original_loss, original_results = self.optimize_body(global_orient.detach(), body_pose, betas, camera_translation,
                                                             camera_center, joints_2d, joints_conf)

        return original_results

    def optimize_body(self, global_orient, body_pose, betas, camera_translation, camera_center, joints_2d, joints_conf):
        """
        Optimize only the body pose and global orientation of the body
        """
        batch_size = global_orient.shape[0]
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = False
        if self.prior_name == 'generator':
            self.latents.requires_grad = True
            body_opt_params = [
                {'params': self.latents, 'lr': self.step_size},
                {'params': betas, 'lr': 1e-2},
                {'params': global_orient, 'lr': 2e-2}
            ]
        else:
            body_pose.requires_grad = True
            body_opt_params = [
                {'params': body_pose, 'lr': self.step_size},
                {'params': betas, 'lr': 1e-2},
                {'params': global_orient, 'lr': 2e-2}
            ]
        body_optimizer = torch.optim.Adam(body_opt_params, betas=(0.9, 0.999))

        stage_weights = [dict(zip(self.loss_weights.keys(), vals)) for vals in zip(*self.loss_weights.values())]

        # for stage, current_weights in enumerate(tqdm(stage_weights, desc='Stage')):
        for stage, current_weights in enumerate(stage_weights):
            for i in range(self.num_iters):
                if self.prior_name == 'generator':
                    # normed_latents = self.latents / self.latents.norm(dim=1, keepdim=True)
                    body_pose = self.generator(self.latents, output_type='aa').view(-1, num_joints*3)
                smpl_output = self.smpl(betas=betas,
                                        body_pose=body_pose,
                                        global_orient=global_orient,
                                        pose2rot=True,
                                        transl=camera_translation)

                model_joints = smpl_output.joints
                t = self.sample_continuous_time(iteration=stage * self.num_iters + i)
                loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                         joints_2d, joints_conf, self.pose_prior, t=t,
                                         focal_length=self.focal_length,
                                         **current_weights, verbose=False, part='body')
                # if self.prior_name == 'generator':
                #     loss = loss + self.latent_prior(self.latents)
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

                # if self.prior_name == 'generator':
                #     with torch.no_grad():
                #         self.latents = self.latents / self.latents.norm(dim=1, keepdim=True)

        # Get final loss value
        with torch.no_grad():
            smpl_output = self.smpl(betas=betas,
                                    body_pose=body_pose,
                                    global_orient=global_orient,
                                    pose2rot=True,
                                    transl=camera_translation)

            model_joints = smpl_output.joints
            t = self.sample_continuous_time(iteration=stage * self.num_iters + i)
            reprojection_loss = body_fitting_loss(body_pose, betas, model_joints, camera_translation, camera_center,
                                                  joints_2d, joints_conf, self.pose_prior, t=t,
                                                  focal_length=self.focal_length,
                                                  output='reprojection', verbose=False, part='body')

        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()
        return reprojection_loss, (pose, betas, camera_translation, reprojection_loss)
