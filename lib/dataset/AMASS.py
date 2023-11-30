import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from lib.utils.transforms import axis_angle_to_rot6d, rot6d_to_axis_angle
from lib.body_model.utils import BodyPartIndices, BodySegIndices

N_POSES = 21


class AMASSDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, version='version0', subset='train',
                 sample_interval=None, rot_rep='rot6d', return_shape=False,
                 normalize=True, min_max=True):

        self.root_path = root_path
        self.version = version
        assert subset in ['train', 'valid', 'test']
        self.subset = subset
        self.sample_interval = sample_interval
        assert rot_rep in ['axis', 'rot6d']
        self.rot_rep = rot_rep
        self.return_shape = return_shape
        self.normalize = normalize
        self.min_max = min_max

        self.poses, self.shapes = self.read_data()

        if self.sample_interval:
            self._sample(sample_interval)
        if self.normalize:
            if self.min_max:
                self.min_poses, self.max_poses, self.min_shapes, self.max_shapes = self.Normalize()
            else:
                self.mean_poses, self.std_poses, self.mean_shapes, self.std_shapes = self.Normalize()

        self.real_data_len = len(self.poses)

    def __getitem__(self, idx):
        """
        Return:
            [21, 3] or [21, 6] for poses including body and root orient
            [10] for shapes (betas)  [Optimal]
        """
        data_poses = self.poses[idx % self.real_data_len]
        data_dict = {'poses': data_poses}
        if self.return_shape:
            data_dict['shapes'] = self.shapes[idx % self.real_data_len]
        return data_dict

    def __len__(self, ):
        return len(self.poses)

    def _sample(self, sample_interval):
        print(f'Class AMASSDataset({self.subset}): sample dataset every {sample_interval} frame')
        self.poses = self.poses[::sample_interval]

    def read_data(self):
        data_path = os.path.join(self.root_path, self.version, self.subset)
        # root_orient = torch.load(os.path.join(data_path, 'root_orient.pt'))
        poses = torch.load(os.path.join(data_path, 'pose_body.pt'))
        shapes = torch.load(os.path.join(data_path, 'betas.pt')) if self.return_shape else None
        # poses = torch.cat([root_orient, pose_body], dim=1)
        data_len = len(poses)
        if self.rot_rep == 'rot6d':
            poses = axis_angle_to_rot6d(poses.reshape(-1, 3)).reshape(data_len, -1)

        return poses, shapes

    def Normalize(self):
        # Use train dataset for normalize computing, Z_score or min-max Normalize
        if self.min_max:
            normalize_path = os.path.join(self.root_path, self.version, 'train', self.rot_rep + '_normalize1.pt')
        else:
            normalize_path = os.path.join(self.root_path, self.version, 'train', self.rot_rep + '_normalize2.pt')

        if os.path.exists(normalize_path):
            normalize_params = torch.load(normalize_path)
            if self.min_max:
                min_poses, max_poses, min_shapes, max_shapes = (
                    normalize_params['min_poses'],
                    normalize_params['max_poses'],
                    normalize_params['min_shapes'],
                    normalize_params['max_shapes']
                )
            else:
                mean_poses, std_poses, mean_shapes, std_shapes = (
                    normalize_params['mean_poses'],
                    normalize_params['std_poses'],
                    normalize_params['mean_shapes'],
                    normalize_params['std_shapes']
                )
        else:
            if self.min_max:
                min_poses = torch.min(self.poses, dim=0)[0]
                max_poses = torch.max(self.poses, dim=0)[0]

                min_shapes = torch.min(self.shapes, dim=0)[0] if self.return_shape else None
                max_shapes = torch.max(self.shapes, dim=0)[0] if self.return_shape else None

                torch.save({
                    'min_poses': min_poses,
                    'max_poses': max_poses,
                    'min_shapes': min_shapes,
                    'max_shapes': max_shapes
                }, normalize_path)
            else:
                mean_poses = torch.mean(self.poses, dim=0)
                std_poses = torch.std(self.poses, dim=0)

                mean_shapes = torch.mean(self.shapes, dim=0) if self.return_shape else None
                std_shapes = torch.std(self.shapes, dim=0) if self.return_shape else None

                torch.save({
                    'mean_poses': mean_poses,
                    'std_poses': std_poses,
                    'mean_shapes': mean_shapes,
                    'std_shapes': std_shapes
                }, normalize_path)

        if self.min_max:
            self.poses = 2 * (self.poses - min_poses) / (max_poses - min_poses) - 1
            if self.return_shape:
                self.shapes = 2 * (self.shapes - min_shapes) / (max_shapes - min_shapes) - 1
            return min_poses, max_poses, min_shapes, max_shapes

        else:
            self.poses = (self.poses - mean_poses) / std_poses
            if self.return_shape:
                self.shapes = (self.shapes - mean_shapes) / std_shapes
            return mean_poses, std_poses, mean_shapes, std_shapes


    def Denormalize(self, poses, shapes=None):
        assert len(poses.shape) == 2 or len(poses.shape) == 3  # [b, data_dim] or [t, b, data_dim]

        if self.min_max:
            min_poses = self.min_poses.view(1, -1).to(poses.device)
            max_poses = self.max_poses.view(1, -1).to(poses.device)

            if len(poses.shape) == 3:  # [t, b, data_dim]
                min_poses = min_poses.unsqueeze(0)
                max_poses = max_poses.unsqueeze(0)

            normalized_poses = 0.5 * ((poses + 1) * (max_poses - min_poses) + 2 * min_poses)

            if shapes is not None and self.min_shapes is not None:
                min_shapes = self.min_shapes.view(1, -1).to(shapes.device)
                max_shapes = self.max_shapes.view(1, -1).to(shapes.device)

                if len(shapes.shape) == 3:
                    min_shapes = min_shapes.unsqueeze(0)
                    max_shapes = max_shapes.unsqueeze(0)

                normalized_shapes = 0.5 * ((shapes + 1) * (max_shapes - min_shapes) + 2 * min_shapes)
                return normalized_poses, normalized_shapes
            else:
                return normalized_poses
        else:
            mean_poses = self.mean_poses.view(1, -1).to(poses.device)
            std_poses = self.std_poses.view(1, -1).to(poses.device)

            if len(poses.shape) == 3:  # [t, b, data_dim]
                mean_poses = mean_poses.unsqueeze(0)
                std_poses = std_poses.unsqueeze(0)

            normalized_poses = poses * std_poses + mean_poses

            if shapes is not None and self.mean_shapes is not None:
                mean_shapes = self.mean_shapes.view(1, -1)
                std_shapes = self.std_shapes.view(1, -1)

                if len(shapes.shape) == 3:
                    mean_shapes = mean_shapes.unsqueeze(0)
                    std_shapes = std_shapes.unsqueeze(0)

                normalized_shapes = shapes * std_shapes + mean_shapes
                return normalized_poses, normalized_shapes
            else:
                return normalized_poses

    def eval(self, preds):
        pass


class Posenormalizer:
    def __init__(self, data_path, device='cuda:0', normalize=True, min_max=True, rot_rep=None):
        assert rot_rep in ['rot6d', 'axis']
        self.normalize = normalize
        self.min_max = min_max
        self.rot_rep = rot_rep
        normalize_params = torch.load(os.path.join(data_path, '{}_normalize1.pt'.format(rot_rep)))
        self.min_poses, self.max_poses = normalize_params['min_poses'].to(device), normalize_params['max_poses'].to(device)
        normalize_params = torch.load(os.path.join(data_path, '{}_normalize2.pt'.format(rot_rep)))
        self.mean_poses, self.std_poses = normalize_params['mean_poses'].to(device), normalize_params['std_poses'].to(device)

    def offline_normalize(self, poses, from_axis=False):
        assert len(poses.shape) == 2 or len(poses.shape) == 3  # [b, data_dim] or [t, b, data_dim]
        pose_shape = poses.shape
        if from_axis and self.rot_rep == 'rot6d':
            poses = axis_angle_to_rot6d(poses.reshape(-1, 3)).reshape(*pose_shape[:-1], -1)

        if not self.normalize:
            return poses

        if self.min_max:
            min_poses = self.min_poses.view(1, -1)
            max_poses = self.max_poses.view(1, -1)

            if len(poses.shape) == 3:  # [t, b, data_dim]
                min_poses = min_poses.unsqueeze(0)
                max_poses = max_poses.unsqueeze(0)

            normalized_poses = 2 * (poses - min_poses) / (max_poses - min_poses) - 1

        else:
            mean_poses = self.mean_poses.view(1, -1)
            std_poses = self.std_poses.view(1, -1)

            if len(poses.shape) == 3:  # [t, b, data_dim]
                mean_poses = mean_poses.unsqueeze(0)
                std_poses = std_poses.unsqueeze(0)

            normalized_poses = (poses - mean_poses) / std_poses

        return normalized_poses

    def offline_denormalize(self, poses, to_axis=False):
        assert len(poses.shape) == 2 or len(poses.shape) == 3  # [b, data_dim] or [t, b, data_dim]

        if not self.normalize:
            denormalized_poses = poses
        else:
            if self.min_max:
                min_poses = self.min_poses.view(1, -1)
                max_poses = self.max_poses.view(1, -1)

                if len(poses.shape) == 3:  # [t, b, data_dim]
                    min_poses = min_poses.unsqueeze(0)
                    max_poses = max_poses.unsqueeze(0)

                denormalized_poses = 0.5 * ((poses + 1) * (max_poses - min_poses) + 2 * min_poses)

            else:
                mean_poses = self.mean_poses.view(1, -1)
                std_poses = self.std_poses.view(1, -1)

                if len(poses.shape) == 3:  # [t, b, data_dim]
                    mean_poses = mean_poses.unsqueeze(0)
                    std_poses = std_poses.unsqueeze(0)

                denormalized_poses = poses * std_poses + mean_poses

        if to_axis and self.rot_rep == 'rot6d':
            pose_shape = denormalized_poses.shape
            denormalized_poses = rot6d_to_axis_angle(denormalized_poses.reshape(-1, 6)).reshape(*pose_shape[:-1], -1)

        return denormalized_poses


# a simple eval process for completion task
class Evaler:
    def __init__(self, body_model, part=None):
        self.body_model = body_model
        self.part = part

        if self.part is not None:
            self.joint_idx = np.array(getattr(BodyPartIndices, self.part)) + 1  # skip pelvis
            self.vert_idx = np.array(getattr(BodySegIndices, self.part))
        else:
            self.joint_idx = slice(None)
            self.vert_idx = slice(None)

    def eval_bodys(self, outs, gts):
        '''
        :param outs: [b, j*3] axis-angle results of body poses
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict for every sample
        '''
        sample_num = len(outs)
        eval_result = {'mpvpe_all': [], 'mpjpe_body': []}
        body_gt = self.body_model(pose_body=gts)
        body_out = self.body_model(pose_body=outs)

        for n in range(sample_num):
            # MPVPE from all vertices
            mesh_gt = body_gt.v.detach().cpu().numpy()[n, self.vert_idx]
            mesh_out = body_out.v.detach().cpu().numpy()[n, self.vert_idx]
            eval_result['mpvpe_all'].append(np.sqrt(np.sum((mesh_out - mesh_gt) ** 2, 1)).mean() * 1000)

            joint_gt_body = body_gt.Jtr.detach().cpu().numpy()[n, self.joint_idx]
            joint_out_body = body_out.Jtr.detach().cpu().numpy()[n, self.joint_idx]

            eval_result['mpjpe_body'].append(
                np.sqrt(np.sum((joint_out_body - joint_gt_body) ** 2, 1)).mean() * 1000)

        return eval_result

    def multi_eval_bodys(self, outs, gts):
        '''
        :param outs: [b, hypo, j*3] axis-angle results of body poses, multiple hypothesis
        :param gts:  [b, j*3] axis-angle groundtruth of body poses
        :return: result dict
        '''
        hypo_num = outs.shape[1]
        eval_result = {f'mpvpe_all': [], f'mpjpe_body': []}
        for hypo in range(hypo_num):
            result = self.eval_bodys(outs[:, hypo], gts)
            eval_result['mpvpe_all'].append(result['mpvpe_all'])
            eval_result['mpjpe_body'].append(result['mpjpe_body'])

        eval_result['mpvpe_all'] = np.min(eval_result['mpvpe_all'], axis=0)
        eval_result['mpjpe_body'] = np.min(eval_result['mpjpe_body'], axis=0)

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print('MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))

    def print_multi_eval_result(self, eval_result, hypo_num):
        print(f'multihypo {hypo_num} MPVPE (All): %.2f mm' % np.mean(eval_result['mpvpe_all']))
        print(f'multihypo {hypo_num} MPJPE (Body): %.2f mm' % np.mean(eval_result['mpjpe_body']))
