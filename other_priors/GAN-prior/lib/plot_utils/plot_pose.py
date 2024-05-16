import torch
import numpy as np
import matplotlib.pyplot as plt

import os
from lib.models.rodrigues_utils import matrot2aa
NUM_SHAPE_SMPL_PARAMS = 10
from lib.functional.renderer.render import run_rendering
from .plot_tsne import compose_features, get_embed, plot_dots


def get_d_real(discr, dl, num_features):
    arr_d_full = {discr_type : torch.empty((0)) for discr_type in ['poseeach','poseall']}
    arr_x_pose = torch.empty((0))

    for sample in dl:
        pose = sample['pose']

        with torch.no_grad():
            d = discr(pose, input_type='aa') # dictionary of different discrs outputs
        
        for discr_type in d:
            arr_d_full[discr_type] = torch.cat((arr_d_full[discr_type], d[discr_type].cpu()))
        arr_x_pose = torch.cat((arr_x_pose, pose))

        if len(arr_x_pose) >= num_features:
            for discr_type in d:
                arr_d_full[discr_type] = arr_d_full[discr_type][:num_features]
            arr_x_pose = arr_x_pose[:num_features]
            break
    
    return arr_d_full, arr_x_pose.flatten(start_dim=1)


def get_d_fake(gen, discr, num_features, batch_size=1_000, num_pose=69, latent_space_dim=100, latent_space_type='S'):

    from lib.functional.sampling import generate_random_input as generate_random_input_
    generate_random_input = lambda batch_size : generate_random_input_(batch_size, latent_space_dim, latent_space_type)
    
    arr_d_full = {discr_type : torch.empty((0)) for discr_type in ['poseeach','poseall']}
    arr_x_pose = torch.empty((0))

    while True:
        z = generate_random_input(batch_size)
        with torch.no_grad():
            x = gen(z, output_type='matrot')
            d = discr(x, input_type='matrot')
            pose = matrot2aa(x.view([-1, 1, x.shape[-2], 9])).squeeze(1).cpu()

        for discr_type in d:
            arr_d_full[discr_type] = torch.cat((arr_d_full[discr_type], d[discr_type].cpu()))
        arr_x_pose = torch.cat((arr_x_pose, pose))

        if len(arr_x_pose) >= num_features:
            for discr_type in d:
                arr_d_full[discr_type] = arr_d_full[discr_type][:num_features]
            arr_x_pose = arr_x_pose[:num_features]
            break

    return arr_d_full, arr_x_pose.flatten(start_dim=1)


def plot_d_predictions(d_real, d_fake, savepath, add_name):

    d_real_poseall = d_real['poseall']
    d_fake_poseall = d_fake['poseall']

    d_real_poseeach = d_real['poseeach']
    d_fake_poseeach = d_fake['poseeach']
    
    os.makedirs(savepath, exist_ok=True)
    add_name = f'{add_name}.png'
    
    for discr_name, arr_real, arr_fake in zip(
                                            ['all_pose', 'pose'],
                                            [d_real_poseall, d_real_poseeach],   
                                            [d_fake_poseall, d_fake_poseeach]
                                            ):

        os.makedirs(f'{savepath}/{discr_name}', exist_ok=True)
        if discr_name != 'pose':
            title = discr_name
            full_img_save_path = f'{savepath}/{discr_name}/{add_name}'
            _plot_and_save_one_discr(arr_real, arr_fake, title, full_img_save_path)
        else:
            ### discr_name == 'pose'
            for idx, (arr_real_, arr_fake_) in enumerate(zip(arr_real.transpose(0,1), arr_fake.transpose(0,1)), start=1):
                os.makedirs(f'{savepath}/{discr_name}/{idx}', exist_ok=True)
                full_img_save_path = f'{savepath}/{discr_name}/{idx}/{add_name}'
                _plot_and_save_one_discr(arr_real_, arr_fake_, f'{discr_name}_{idx}', full_img_save_path)


def _plot_and_save_one_discr(arr_real, arr_fake, title, full_img_save_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(14,7))
    ax.set_title(f'{title}', fontsize='xx-large')
    min_val = np.min( [np.min(arr_real.numpy()), np.min(arr_fake.numpy())] )
    max_val = np.max( [np.max(arr_real.numpy()), np.max(arr_fake.numpy())] )
    
    ax.hist(arr_real.numpy(), bins=300, range=(min_val,max_val), color='yellow', label='real', alpha=0.5)
    ax.hist(arr_fake.numpy(), bins=300, range=(min_val,max_val), color='lightblue', label='fake', alpha=0.5)
    ax.legend(fontsize='xx-large')
    ax.set_xlim((-0.5, 1.5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([0., 0.5, 1.])
    ax.xaxis.set_ticklabels([0., 0.5, 1.])
    ax.tick_params(axis='x', which='both', length=0)

    plt.subplots_adjust(left=0.001, right=0.999, top=0.9, bottom=0.1)

    fig.savefig(full_img_save_path)
    plt.close()


def plot_tsne(reals, fakes, savepath, num_features=1_000, add_name='', seed=None, verbose=0):

    os.makedirs(f'{savepath}', exist_ok=True)

    mode_ = name = 'pose'
    x, masks = compose_features([reals, fakes], num_features, seed=seed)
    x_embed = get_embed(x, embed_alg='TSNE', dim=2, verbose=verbose)

    fig = plot_dots(x_embed, masks, name=name, labels=['real', 'fake'])
    os.makedirs(f'{savepath}/{mode_}', exist_ok=True)
    fig.savefig(f'{savepath}/{mode_}/{add_name}.png')


def plot_poses(trainer, poses, savepath, add_name='', img_size=512, cam_distance=2.4, batch_size_max=100):
    os.makedirs(f'{savepath}', exist_ok=True)
    num_samples = 16

    ids = list(range(len(poses)))
    np.random.shuffle(ids)
    assert len(ids) >= num_samples, 'Check the size of the ds array!'
    ids = ids[:num_samples]

    ### check for body pose size: 21 - SMPL-H (for VPoser), 23 - SMPL
    if poses.shape[1] == 21*3:
        poses = torch.cat([poses, torch.zeros(len(poses), 6).to(device=poses.device)], dim=1)
    
    X = torch.cat([poses[ids].to(device=trainer.device0), torch.zeros([num_samples, NUM_SHAPE_SMPL_PARAMS], device=trainer.device0)], dim=1)
    images = run_rendering(X, img_size=img_size, cam_distance=cam_distance, device=trainer.device0,
                           model_type='smpl', model_folder=trainer.cfg.SMPL_PATH, batch_size_max=batch_size_max)
    
    fig, ax = plt.subplots(4, 4, figsize=(14, 14))
    axs = ax.reshape(-1)
    for i in range(len(images)):
        axs[i].imshow(images[i])
        axs[i].set_axis_off()
    fig.savefig(f'{savepath}/{add_name}.png')
    plt.close()
    