import sys
sys.path.append('../')

from tqdm import tqdm

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_features_from_dataloader(dl, num_features=10_000):
    arr_full, arr_pose, arr_shape = torch.empty((0)), torch.empty((0)), torch.empty((0))

    for sample in tqdm(dl):
        pose = sample['smpl_pose'] 
        shape = sample['smpl_shape']
        x = torch.cat((pose, shape), dim=1)

        arr_full = torch.cat((arr_full, x))
        arr_pose = torch.cat((arr_pose, pose))
        arr_shape = torch.cat((arr_shape, shape))

        if len(arr_full) >= num_features:
            arr_full = arr_full[:num_features]
            arr_pose = arr_pose[:num_features]
            arr_shape = arr_shape[:num_features]
            return arr_full, arr_pose, arr_shape 
        

def get_features_from_latent(generator, num_features=10_000, batch_size=1_000, latent_space_dim=100, latent_space_type='S', num_pose=69, device='cpu'):
    
    from lib.functional.sampling import generate_random_input as generate_random_input_
    generate_random_input = lambda batch_size : generate_random_input_(batch_size, latent_space_dim, latent_space_type)
    
    arr_full, arr_pose, arr_shape = torch.empty((0)), torch.empty((0)), torch.empty((0))

    while True:
        z = generate_random_input(batch_size)
        z = z.to(device=device)
        with torch.no_grad():
            x = generator(z).cpu()
            pose, shape = x[:,:num_pose], x[:,num_pose:]

        arr_full = torch.cat((arr_full, x))
        arr_pose = torch.cat((arr_pose, pose))
        arr_shape = torch.cat((arr_shape, shape))

        if len(arr_full) >= num_features:
            arr_full = arr_full[:num_features]
            arr_pose = arr_pose[:num_features]
            arr_shape = arr_shape[:num_features]
            return arr_full, arr_pose, arr_shape 


def compose_features(arrs, num_samples, seed=1234):
    res = torch.empty((0))
    if seed is not None:
        np.random.seed(seed)
    for arr in arrs:
        ids = list(range(len(arr)))
        np.random.shuffle(ids)
        assert len(ids) > num_samples, 'Check the size of the ds array!'
        ids = ids[:num_samples]
        res = torch.cat((res, arr[ids]))
    
    # now random samples from different domains are stacked together.
    # Let's shuffle them and save the mask:
    final_ids = np.arange(len(res))
    np.random.shuffle(final_ids)
    res = res[final_ids]
    
    masks = []
    prev = 0
    for arr in arrs:
        mask = (final_ids < num_samples+prev) & (prev <= final_ids)
        masks.append(mask)
        prev += num_samples
    
    return res, masks


def get_embed(x, embed_alg, dim, verbose=0):
    import sklearn.decomposition, sklearn.manifold

    if embed_alg == 'PCA':
        x_embed = sklearn.decomposition.PCA(dim).fit(x).transform(x)
    elif embed_alg == 'TSNE':
        x_embed = sklearn.manifold.TSNE(dim, init='pca', n_iter=1000, n_jobs=-1, verbose=verbose).fit_transform(x)

    return x_embed



def plot_dots(feats, masks, name='', labels=None, show=False):
    
    plt.style.use('dark_background')
    colors = ['yellow', 'lightblue', 'r', 'w', 'c', 'm', 'b', 'k'][:len(masks)]
    fig, ax = plt.subplots(1, 1, figsize=(14,14))
    if labels is None:
        labels = [f'mask{i}' for i in range(len(masks))]
        
    for mask, color, label in zip(masks, colors, labels):
        ax.plot(feats[mask,0], feats[mask,1], '.', c=color, markersize=4, label=label, alpha=1)
        
    ax.set_title(f'{name}', fontsize='xx-large')
    ax.set_axis_off()
    ax.axis('equal')
    # ax.set_aspect('equal', 'box')
    ax.legend(fontsize='xx-large')
    fig.tight_layout()

    if show:
        plt.show(block=False) # block == True would freeze code state at this interactive figure
                              # block == False shows figure in non-interactive mode and continues code execution
        plt.pause(1)
        plt.show(block=True)
    else:
        return fig


def plot_tsne(reals, fakes, savepath, num_features=1_000, add_name='', seed=None, verbose=0):

    os.makedirs(f'{savepath}', exist_ok=True)

    x_real_full, x_real_pose, x_real_shape = reals
    x_fake_full, x_fake_pose, x_fake_shape = fakes

    for mode_, real_feat, fake_feat in zip(
                                    ['full', 'pose', 'shape'], 
                                    [x_real_full, x_real_pose, x_real_shape],
                                    [x_fake_full, x_fake_pose, x_fake_shape],
                                            ):
        
        name = f'{mode_}'
        x, masks = compose_features([real_feat, fake_feat], num_features, seed=seed)
        x_embed = get_embed(x, embed_alg='TSNE', dim=2, verbose=verbose)

        fig = plot_dots(x_embed, masks, name=name, labels=['real', 'pred'])
        os.makedirs(f'{savepath}/{mode_}', exist_ok=True)
        fig.savefig(f'{savepath}/{mode_}/{add_name}.png')
