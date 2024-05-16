import torch
import numpy as np

import pytorch3d.structures
import pytorch3d.renderer

def render(smpl_vec, renderer, smpl_model, faces, device, model_type='smpl', background_white=False):
    '''
    Generates rendered images given a vector of SMPL parameters. Images are colored in normals as RGB.
    Both renderer and SMPL model must also be provided.
    Parameters:
    smpl_vec (tensor, B x N) - vector of SMPL parameters, N = Npose+Nshape
    renderer - Renderer class instance
    smpl_model - SMPL class instance, loaded model
    faces - faces indices of SMPL model

    Returns:
    images (numpy array, B x 3 x img_size x img_size)
    '''

    batch_size = smpl_vec.size(0)

    if model_type == 'smpl':
        num_pose = 69
    elif model_type == 'smplx':
        num_pose = 63

    pose, shape = smpl_vec[:,:num_pose], smpl_vec[:,num_pose:]
    output = smpl_model(betas=shape, body_pose=pose, expression=None, return_verts=True)
    vertices = output.vertices
    vertices = vertices.reshape(batch_size, vertices.shape[-2], 3) # B x Ntri x 3

    faces = torch.tensor(faces.copy()).unsqueeze(0).repeat(batch_size,1,1).to(device)
    meshes = pytorch3d.structures.Meshes(vertices, faces)

    verts_shape = meshes.verts_packed().shape
    verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
    meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)

    with torch.no_grad():
        normal_out, silhouette_out = renderer(meshes)

    images = (255*normal_out.cpu().numpy()).astype(np.uint8)
    if background_white:
        images[images == 0] = 255 # change background to white
    return images


def render_silhouettes(smpl_vec, renderer, smpl_model, faces, device=torch.device('cuda:0'), model_type='smpl'):
    '''
    Generates rendered images given a vector of SMPL parameters. Images are colored in normals as RGB.
    Both renderer and SMPL model must also be provided.
    Parameters:
    smpl_vec (tensor, B x N) - vector of SMPL parameters, N = Npose+Nshape
    renderer - Renderer class instance
    smpl_model - SMPL class instance, loaded model
    faces - faces indices of SMPL model

    Returns:
    images (numpy array, B x 3 x img_size x img_size)
    '''

    batch_size = smpl_vec.size(0)

    if model_type == 'smpl':
        num_pose = 69
    elif model_type == 'smplx':
        num_pose = 63

    ### render silhouettes
    pose, shape = smpl_vec[:,:num_pose], smpl_vec[:,num_pose:]
    output = smpl_model(betas=shape, body_pose=pose, expression=None, return_verts=True)
    vertices = output.vertices
    vertices = vertices.reshape(batch_size, vertices.shape[-2], 3) # B x Ntri x 3

    faces = torch.tensor(faces.copy()).unsqueeze(0).repeat(batch_size,1,1).to(device)

    ### MUST BE DIFFERENTIABLE!!!
    silhouette_out = renderer(vertices, faces)
    
    return silhouette_out


def add_frames(seq, add_start=0, add_end=0, add_reverse=False):
    '''
    seq (tensor, batch_size x n_frames x dim_size) - a sequence of frames / smpl vectors / latent vectors

    add_start (int) - adds the first element <add_start> times in the beginning
    add_end (int) - adds the last element <add_end> times in the end
    add_reverse (bool) - if True, reverses the seq (after add_start and add_end) and adds it to the the end

    NOTE The tensor should not lie on any device, as <add_reverse> requires the casting to numpy
    TODO Putting the tensor back onto device is not implemented yet.
    '''
    assert add_start >= 0 and add_end >= 0

    batch_size, n_frames, dim_size = seq.shape

    if add_start > 0:
        add_ = (seq[:,0].unsqueeze(1)).repeat(1,add_start,1)
        seq = torch.cat([add_, seq], dim=1)

    if add_end > 0:
        add_ = (seq[:,-1].unsqueeze(1)).repeat(1,add_end,1)
        seq = torch.cat([seq, add_], dim=1)
    
    if add_reverse: 
        seq = torch.from_numpy(np.concatenate((seq.numpy(), seq.numpy()[:,::-1]), axis=1))

    return seq


def init_smpl(batch_size, device, model_folder, model_type='smpl'):
    import smplx
    if model_type == 'smpl':
        model_folder = model_folder
        ext = None
        num_betas = 10
    else:
        raise NotImplementedError
    model_folder = '/data3/ljz24/projects/3d/body_models/smpl/SMPL_NEUTRAL.pkl'
    smpl_model = smplx.create(model_folder, model_type=model_type, create_global_orient=True, batch_size=batch_size, num_betas=num_betas, ext=ext).to(device)
    faces = smpl_model.faces.astype(int)
    return smpl_model, faces



def run_rendering(smpl_vec, img_size=512, cam_distance=2.4, device=torch.device('cuda:0'),
                  model_type='smpl', model_folder=None, batch_size_max=20, background_white=False):
    batch_size = smpl_vec.size(0)

    ### split in chunks if necessary
    if batch_size > batch_size_max:
        images = []
        images_total_done = 0
        for smpl_vec_ in torch.split(smpl_vec, batch_size_max):
            images_ = run_rendering(smpl_vec_, img_size, cam_distance, device, model_type=model_type,
                                    model_folder=model_folder, background_white=background_white)
            images.append(images_)
            images_total_done += len(images_)
            print(images_total_done, 'out of ', len(smpl_vec))
        images = np.concatenate(images, axis=0)
        return images

    ### initialize renderer
    from .utils import get_renderer
    renderer = get_renderer(batch_size, img_size, cam_distance, device)

    ### initialize SMPL model
    smpl_model, faces = init_smpl(batch_size, device, model_type=model_type, model_folder=model_folder)

    ### generate images
    images = render(smpl_vec.to(device), renderer, smpl_model, faces, device, model_type=model_type, background_white=background_white)

    return images