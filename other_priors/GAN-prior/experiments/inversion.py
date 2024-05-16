import torch
import torch.optim as optim
import numpy as np
from lib.models.gan import Generator

device = "cuda:0"
### load the model
latent_size = 32
num_joints = 21

net = Generator(latentD=latent_size, num_joints=num_joints)

ckpt = './output/ganS/gan_S_amass/ckpt_0300.pth'
ckpt = torch.load(ckpt, map_location='cpu')['generator_state_dict']
net.load_state_dict(ckpt)
net.eval()
net = net.to(device)

file_path = '/data3/ljz24/projects/3d/DPoser/examples/toy_body_data.npz'
data = np.load(file_path, allow_pickle=True)
sample_num = 100
target_pose_parameters = data['pose_samples'][:sample_num]
target_pose_parameters = torch.from_numpy(target_pose_parameters).to(device)

# Initialize a latent vector
latent_vector = torch.randn(sample_num, latent_size).to(device)
latent_vector.requires_grad = True

# Choose an optimizer
optimizer = optim.Adam([latent_vector], lr=0.1)

# Define the number of iterations for the inversion process
num_iterations = 1000

for iteration in range(num_iterations):
    optimizer.zero_grad()

    # Generate pose parameters from the latent vector
    generated_pose = net(latent_vector, output_type='aa').reshape(sample_num, -1)

    # Compute the loss (e.g., MSE)
    loss = torch.nn.functional.mse_loss(generated_pose, target_pose_parameters)

    # Backpropagation
    loss.backward()
    optimizer.step()

    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")

"""
RUN:
python -m experiments.inversion
"""