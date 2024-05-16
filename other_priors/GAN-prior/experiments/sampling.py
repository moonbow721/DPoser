import torch
from lib.models.gan import Generator
from lib.functional.sampling import generate_random_input

device = "cuda:0"
### load the model
latent_size = 32
num_joints = 21
batch_size = 64

net = Generator(latentD=latent_size, num_joints=num_joints)

ckpt = './output/ganS/gan_S_amass/ckpt_0300.pth'
ckpt = torch.load(ckpt, map_location='cpu')['generator_state_dict']
net.load_state_dict(ckpt)
net.eval()
net = net.to(device)

### sample random latent points
seed = 42
torch.manual_seed(seed)
batch_size = 50000
z = generate_random_input(batch_size, latent_size, latent_space_type='S', device=device)
print(z.shape)

sampled_pose_body = net(z.view(-1,latent_size), output_type='aa').view(-1, num_joints*3) # batch_size x 3*num_joints
torch.save(sampled_pose_body.detach().cpu(), '/data3/ljz24/projects/3d/DPoser/output/outer_samples/gan_samples.pt')
print(sampled_pose_body.shape)


"""
RUN:
python -m experiments.sampling
"""