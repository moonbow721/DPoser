import os
import sys

import cv2
import numpy as np
import torch
from absl import app
from absl.flags import argparse_flags
from tqdm import tqdm

sys.path.insert(0, '/data3/ljz24/projects/3d')
sys.path.insert(0, '/data3/ljz24/projects/3d/human_body_prior/src')
sys.path.insert(0, '/data3/ljz24/projects/3d/DPoser')
from os import path as osp
from lib.body_model.body_model import BodyModel
from lib.body_model.visual import save_obj, render_mesh, faster_render
from lib.body_model.body_model import BodyModel

bg_img = np.ones([512, 384, 3]) * 255  # background canvas
focal = [1500, 1500]
princpt = [200, 192]


def parse_args(argv):
    parser = argparse_flags.ArgumentParser(description='VPoser motion denoising')

    parser.add_argument('--bodymodel-path', type=str,
                        default='../body_models/smplx/SMPLX_NEUTRAL.npz',
                        help='load SMPLX')
    parser.add_argument('--outpath-folder', type=str, default='./vposer_denoising/40noise')
    parser.add_argument('--noise-std', type=float, default=0.04, help='control added noise')
    parser.add_argument('--device', type=str, default='cuda:0')

    # data preparation
    parser.add_argument('--file-path', type=str, help='use toy data to run')
    parser.add_argument('--dataset', type=str, default='AMASS')

    args = parser.parse_args(argv[1:])

    return args


class MotionDenoise(object):
    def __init__(self, args, model, body_model, out_path=None, debug=False, batch_size=1):
        self.args = args
        self.debug = debug
        self.device = args.device
        self.body_model = body_model
        self.out_path = out_path  # only needed for visualization
        self.batch_size = batch_size
        self.betas = torch.zeros((batch_size, 10), device=self.device)
        self.poses = torch.randn((batch_size, 63), device=self.device) * 0.10
        self.VAE = model

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'temp': lambda cst, it: 10. ** 1 * cst * (1 + it),
                       'data': lambda cst, it: 10. ** 2 * cst / (1 + it * it),
                       'vposer': lambda cst, it: 0.01 * cst * (1 + it)
                       }
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    @staticmethod
    def visualize(vertices, faces, out_path, render=False, prefix='out', save_mesh=False, faster=False, device=None):
        # save meshes and rendered results if needed
        os.makedirs(out_path, exist_ok=True)
        if save_mesh:
            vertices = vertices.detach().cpu()
            faces = faces.cpu()
            os.makedirs(os.path.join(out_path, 'meshes'), exist_ok=True)
            [save_obj(vertices[i], faces, os.path.join(out_path, 'meshes', '{}_{:04}.obj'.format(prefix, i))) for i in
             range(len(vertices))]

        if render:
            os.makedirs(os.path.join(out_path, 'renders'), exist_ok=True)
            if faster:
                assert device is not None
                target_path = os.path.join(out_path, 'renders')
                faster_render(vertices, faces, target_path, prefix + '_{:04}.jpg', device)
            else:
                vertices = vertices.detach().cpu()
                faces = faces.cpu()
                for i in range(len(vertices)):
                    rendered_img = render_mesh(bg_img, vertices[i], faces, {'focal': focal, 'princpt': princpt},
                                               view='front')
                    cv2.imwrite(os.path.join(out_path, 'renders', '{}_{:04}.png'.format(prefix, i)), rendered_img)

    def optimize(self, joints3d, gt_poses, iterations=5, steps_per_iter=50, verbose=False, vis=False):
        # create initial SMPL joints and vertices for visualition(to be used for data term)
        smpl_init = self.body_model(betas=self.betas, body_pose=self.poses)
        smpl_gt = self.body_model(betas=self.betas, body_pose=gt_poses)
        if vis:
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, render=True, prefix='init', faster=True,
                           device=self.device)
            self.visualize(smpl_gt.v, smpl_gt.f, self.out_path, render=True, prefix='gt', faster=True,
                           device=self.device)

        joint_error = joints3d - smpl_gt.Jtr[:, :22]
        init_MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 100.
        if verbose:
            print('before denoising:{:0.8f} cm'.format(init_MPJPE.mean()))

        init_joints = joints3d.detach()

        # Optimizer
        smpl_init.body_pose.requires_grad = True
        optimizer = torch.optim.Adam([smpl_init.body_pose], 0.03, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            if verbose:
                loop = tqdm(range(steps_per_iter))
                loop.set_description('Optimizing SMPL poses')
            else:
                loop = range(steps_per_iter)
            for i in loop:
                optimizer.zero_grad()
                loss_dict = dict()

                poses = smpl_init.body_pose
                encoding = self.VAE.encode(poses).mean
                l2_norm_per_sample = torch.norm(encoding, p=2, dim=[1], keepdim=True)
                l2_norm_average = torch.mean(l2_norm_per_sample)
                loss_dict['vposer'] = l2_norm_average

                # calculate temporal loss between mesh vertices
                smpl_init = self.body_model(betas=smpl_init.betas, body_pose=smpl_init.body_pose)
                temp_term = smpl_init.v[:-1] - smpl_init.v[1:]
                loss_dict['temp'] = torch.mean(torch.sqrt(torch.sum(temp_term * temp_term, dim=2)))

                # calculate data term from inital noisy pose
                data_term = smpl_init.Jtr[:, :22] - init_joints
                data_term = torch.mean(torch.sqrt(torch.sum(data_term * data_term, dim=2)))
                if data_term > 0:  # for nans
                    loss_dict['data'] = data_term

                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()
                if verbose:
                    # only for check
                    joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
                    joint_error = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2))) * 100.

                    l_str = 'Step: {} Iter: {}'.format(it, i)
                    l_str += ' j2j : {:0.8f}'.format(joint_error)
                    l_str += ' total : {:0.8f}'.format(tot_loss)
                    for k in loss_dict:
                        l_str += ', {}: {:0.8f}'.format(k, loss_dict[k].mean().item())
                    loop.set_description(l_str)

        smpl_init = self.body_model(betas=self.betas, body_pose=smpl_init.body_pose)
        if vis:
            self.visualize(smpl_init.v, smpl_init.f, self.out_path, render=True, prefix='out', faster=False,
                           device=self.device)

        joint_error = smpl_init.Jtr[:, :22] - smpl_gt.Jtr[:, :22]
        vert_error = smpl_init.v - smpl_gt.v
        MPJPE = torch.mean(torch.sqrt(torch.sum(joint_error * joint_error, dim=2)), dim=1) * 100.  # remain batch dim
        MPVPE = torch.mean(torch.sqrt(torch.sum(vert_error * vert_error, dim=2)), dim=1) * 100.
        if verbose:
            print('after denoising:{:0.8f} cm'.format(MPJPE.mean()))
        results_dict = {'init_MPJPE': init_MPJPE.detach().cpu().numpy(), 'MPJPE': MPJPE.detach().cpu().numpy(),
                        'MPVPE': MPVPE.detach().cpu().numpy()}
        return results_dict


def denoise(config, args, model, gt_file, out_path, std=0.04, verbose=False):
    motion_data_gt = np.load(gt_file)['pose_body']
    batch_size = len(motion_data_gt)
    gt_poses = torch.from_numpy(motion_data_gt.astype(np.float32)).to(args.device)  # [batchsize, 63]

    #  load body model
    body_model = BodyModel(bm_path=args.bodymodel_path, model_type='smplx', batch_size=batch_size, num_betas=10).to(
        args.device)

    # generate noise on joints
    joints3d = body_model(body_pose=gt_poses).Jtr[:, :22]
    noisy_joints3d = joints3d + std * torch.randn(*joints3d.shape, device=joints3d.device)

    # create Motion denoiser layer
    motion_denoiser = MotionDenoise(args, model, body_model=body_model, batch_size=batch_size, out_path=out_path)

    if std == 0.02:
        kwargs = {'iterations': 3, 'steps_per_iter': 40, }
    elif std == 0.04:
        kwargs = {'iterations': 3, 'steps_per_iter': 60, }
    elif std == 0.1:
        kwargs = {'iterations': 3, 'steps_per_iter': 80, }
    else:
        raise NotImplementedError()

    if args.file_path is not None:  # visualization for toy data
        verbose = True
        kwargs['vis'] = True

    batch_results = motion_denoiser.optimize(noisy_joints3d, gt_poses, verbose=verbose, **kwargs)
    return batch_results


def main(args):
    # torch.manual_seed(42)
    config = None
    support_dir = '../support_data/dowloads'
    from human_body_prior.tools.model_loader import load_vposer

    expr_dir = osp.join(support_dir, 'vposer_v1_0')
    vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
    vposer_pt = vposer_pt.to(args.device)

    os.makedirs(args.outpath_folder, exist_ok=True)
    batch_results = denoise(config, args, vposer_pt, args.file_path, args.outpath_folder, std=args.noise_std)
    for key, value in batch_results.items():
        average_value = np.mean(np.array(value))
        print(f"The average of {key} is {average_value}")


if __name__ == '__main__':
    app.run(main, flags_parser=parse_args)
    """
    RUN:
    python motion_denoising.py --file-path path_to_Gestures_3_poses_batch005.npz
    """