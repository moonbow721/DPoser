'''
The module provides the implementation of the SLERP procedure (https://en.wikipedia.org/wiki/Slerp).

SLERP
Given x,y - two vectors that lie on a high-dimensional sphere, defining a two-dimensional circle. 
The vector z from this circle can be defined as follows:

z = SLERP (x, y, phi) := x * sin( theta - phi ) / sin( theta )  +  y * sin( phi ) / sin( theta )

where theta - the angle between x and y, phi - the angle of rotation from x to z.

If phi lies in the range [0, theta], then such z are interpolants x -> y.
If phi lies in the range [0, 2 pi],  then such z are points on the full circle x -> y -> -x -> -y -> x.


Also, the module provides linear interpolation.
'''


import torch
import math

def slerp(x, y, steps=10, slerp_type='in'):
    '''
    Creates a tensor of size `steps` whose values are evenly spaced on a 2D circle, defined by `x` and `y`.
    The return sequence keeps points on a generation is done for pairs of `x` and `y` vectors in the batch.
    The number of samples is `steps`.

    z_i := SLERP (`x`, `y`, phi_i)

    It is assumed that all `x` and `y` vectors are normalized by 1 (lie on a unit sphere) # NOTE

    Args:
        x (tensor, B x D) : the starting value for the set of points
        y (tensor, B x D) : the ending value for the set of points
        steps (int) : number of points in the path
        slerp_type (str) : type of the created slerp sequence
            Options: 'in' - interpolation from `x` to `y`,
                     'circle' - extrapolate in a full circle `x` -> `y` -> `x`.

    Return:
        out (tensor, B x steps x D): sequences of spherical interpolations for all pairs of `x`/`y` vectors
    '''
    assert slerp_type in ['in', 'circle']
    assert x.size(0) == y.size(0)

    B = x.size(0)

    cos_theta = (x * y).sum(dim=1, keepdim=True)
    theta = torch.acos(cos_theta).repeat(1,steps).view(B,-1,1) # B x steps x 1

    interp_steps = torch.linspace(0, 1, steps).repeat(B,1).view(B,-1,1).to(x.device) # B x steps x 1
    interp_steps = interp_steps.to(x.device)

    if slerp_type == 'in':
        phi = theta * interp_steps # B x steps x 1
    elif slerp_type == 'circle':
        phi = 2 * math.pi * interp_steps # B x steps x 1

    sin_theta = torch.sin(theta)
    alpha = torch.sin(theta - phi) / sin_theta # B x steps x 1
    beta = torch.sin(phi) / sin_theta # B x steps x 1
    z = alpha * x.view(B,1,-1) + beta * y.view(B,1,-1) # B x steps x D
    return z.transpose(0, 1)


def linear(A, B, frames):
    A, B = A.unsqueeze(0), B.unsqueeze(0)
    alpha = torch.linspace(0, 1, frames, device=A.device)
    while alpha.dim() < A.dim():
        alpha = alpha.unsqueeze(-1)

    interpolated = (1 - alpha) * A + alpha * B
    return interpolated


def create_sequence(x, y, steps, seq_type='S', slerp_type='in'):
    '''
    mode (str) : type of the created sequence. 
            Options: 'S' - SLERP
                     'L' - Linear
                     'LS' - Linear, then reproject on the sphere.
    slerp_type : type of the slerp interpolation. 
            Options: 'in' - interpolation from `x` to `y`
                     'circle' - extrapolate in a full circle `x` -> `y` -> `x`
    '''

    if seq_type == 'S':
        z = slerp(x, y, steps=steps, slerp_type=slerp_type)
        cos_thetas = (x * y).sum(dim=1, keepdim=True)
        thetas = torch.acos(cos_thetas) # B x 1

    elif seq_type == 'L':
        z = linear(x, y, steps)
        thetas = (y - x).norm(dim=1, keepdim=True) # B x 1

    elif seq_type == 'LS':
        z = linear(x, y, steps)
        z = z / z.norm(dim=-1, keepdim=True)
        thetas = (y - x).norm(dim=1, keepdim=True) # B x 1

    else:
        raise ValueError(f'seq_type "{seq_type}" is not valid!')

    return z, thetas


def get_interpolation(x, y, steps=10, latent_space_type='S', **kwargs):
    if latent_space_type == 'S':
        return create_sequence(x, y, steps=steps, **kwargs)[0]
    else:
        return linear(x, y, steps)


def plot_test_sequence(seq_type, steps=10, slerp_type='in', space_dim=2):
    seed = 0
    B = 100
    torch.manual_seed(seed)

    from sampling import generate_random_input as _generate_random_input
    x = _generate_random_input(B, latent_space_dim=space_dim, latent_space_type='S')
    y = _generate_random_input(B, latent_space_dim=space_dim, latent_space_type='S')

    z, _ = create_sequence(x, y, steps, seq_type, slerp_type)

    ######
    import matplotlib.pyplot as plt

    row_max = 2
    col_max = 5

    cell_size = 2.15


    fig,ax = plt.subplots(1,1,figsize=(col_max*2,row_max*2))
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(left=-cell_size/2, right=cell_size*col_max-cell_size/2)
    ax.set_ylim(top=-cell_size/2, bottom=cell_size*row_max-cell_size/2)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    
    for idx, z_steps in enumerate(z): 
        if idx >= row_max * col_max:
            break
        i,j = idx // col_max, idx % col_max
        x0, y0 = cell_size * j, cell_size * i
        circle = plt.Circle((x0,y0), 1, fill=False)
        ax.add_artist(circle)
        
        for step, z_vec in enumerate(z_steps):
            z_x, z_y = z_vec[0], z_vec[1]
            
            if step == 0:
                ax.plot([x0,x0+z_x], [y0,y0+z_y], 'r-')
            elif step == len(z_steps) - 1:
                ax.plot([x0,x0+z_x], [y0,y0+z_y], 'b-')
            else:
                ax.scatter([x0+z_x], [y0 + z_y])


    plt.show(block=False) # block == True would freeze code state at this interactive figure
                          # block == False shows figure in non-interactive mode and continues code execution
    plt.pause(1)

    plt.show(block=True)



if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description="Interpolation")
    arg_parser.add_argument("--seq_type", required=True)
    arg_parser.add_argument("--slerp_type", required=False, default='in')
    arg_parser.add_argument("--steps", required=False, default=10)
    arg_parser.add_argument("--space_dim", required=False, default=2)
    
    args = arg_parser.parse_args()

    seq_type = args.seq_type
    steps = int(args.steps)
    slerp_type = args.slerp_type
    space_dim = int(args.space_dim)
    plot_test_sequence(seq_type, steps=steps, slerp_type=slerp_type, space_dim=space_dim)