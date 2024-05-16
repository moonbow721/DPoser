import torch

def generate_random_input(batch_size, latent_space_dim, latent_space_type='U', device=None):
    if latent_space_type == 'U':
        z = torch.rand((batch_size, latent_space_dim), device=device) * 2 - 1 # z distributed from -1 to 1 uniformly
    elif latent_space_type == 'S':
        z = torch.randn((batch_size, latent_space_dim), device=device)
        z = z / z.norm(dim=1, keepdim=True) # z distributed uniformly on the unit sphere
    elif latent_space_type == 'N':
        z = torch.randn((batch_size, latent_space_dim), device=device)  # z has standard Gaussian distribution
    else:
        raise NotImplementedError
    return z


def generate_uniform_pair(): # TODO
    raise NotImplementedError