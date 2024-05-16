import os.path as osp

import torch

support_dir = '../support_data/dowloads'
num_poses = 50000  # number of body poses in each batch

version = 'v1'
mode = 'generation'
device = 'cuda:0'
output_path = '../rebuttal_vis_results'

torch.manual_seed(42)
if version == 'v1':
    from human_body_prior.tools.model_loader import load_vposer

    expr_dir = osp.join(support_dir, 'vposer_v1_0')
    vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
    vposer_pt = vposer_pt.to(device)
    sampled_pose_body = vposer_pt.sample_poses(num_poses=num_poses).reshape(num_poses, -1)

else:
    expr_dir = osp.join(support_dir, 'vposer_v2_05')

    print(expr_dir)
    # Loading VPoser Body Pose Prior
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser

    vp, ps = load_model(expr_dir, model_code=VPoser,
                        remove_words_in_model_weights='vp_model.',
                        disable_grad=True)
    vp = vp.to(device)
    sampled_pose_body = vp.sample_poses(num_poses=num_poses)['pose_body'].contiguous().view(num_poses, -1)


torch.save(sampled_pose_body.detach().cpu(), './vposer_samples.pt')

"""
RUN:
python sample_poses.py
"""