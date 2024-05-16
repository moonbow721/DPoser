import torch
from torch import nn
from torch.nn import functional as F

from .rodrigues_utils import batch_rodrigues, matrot2aa, ContinousRotReprDecoder

class Generator(nn.Module):
    def __init__(self, num_neurons=512, latentD=32, num_joints=21, use_cont_repr=True):
        super(Generator, self).__init__()

        self.latentD = latentD
        self.num_joints = num_joints
        self.use_cont_repr = use_cont_repr

        self.dropout = nn.Dropout(p=.1, inplace=False)

        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()
        
        self.bodyprior_dec_out = nn.Linear(num_neurons, self.num_joints* 6)

    def forward(self, Zin, output_type='matrot'):
        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=.2)
        Xout = self.dropout(Xout)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=.2)
        Xout = self.bodyprior_dec_out(Xout)
        if self.use_cont_repr:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)
        
        Xout = Xout.view([-1, 1, self.num_joints, 9])
        if output_type == 'aa':
            Xout = matrot2aa(Xout)
        Xout = Xout.squeeze(1)
        return Xout



class PoseEachDiscriminator(nn.Module):
    def __init__(self, num_joints, hidden_poseeach=32):
        super(PoseEachDiscriminator, self).__init__()
        self.num_joints = num_joints
        self.fc_layer = nn.ModuleList()
        for idx in range(self.num_joints):
            self.fc_layer.append(nn.Linear(in_features=hidden_poseeach, out_features=1))

    def forward(self, comm_features):
        # getting individual pose outputs
        # common features is of shape [N x hidden_comm x num_joints]
        d_each = []
        for idx in range(self.num_joints):
            d_each.append(self.fc_layer[idx](comm_features[:,:,idx]))
        d_each_out = torch.cat(d_each, 1) # N x 23
        return d_each_out


class PoseAllDiscriminator(nn.Module):
    def __init__(self, num_joints, hidden_poseall=1024, num_layers=2):
        super(PoseAllDiscriminator, self).__init__()
        self.num_joints = num_joints
        self.hidden_poseall = hidden_poseall
        self.num_layers = num_layers

        fc_all_pose = [
            nn.Linear(in_features=32*self.num_joints, out_features=hidden_poseall), 
            nn.LeakyReLU(0.2),
            ]
        for _ in range(num_layers - 1):
            fc_all_pose += [
                    nn.Linear(in_features=hidden_poseall, out_features=hidden_poseall), 
                    nn.LeakyReLU(0.2),
                    ]
        fc_all_pose += [nn.Linear(in_features=hidden_poseall, out_features=1)]
        self.fc_all_pose = nn.Sequential(*fc_all_pose)

    def forward(self, comm_features):
        # getting pose-all output
        # common features is of shape [N x hidden_comm x num_joints]
        d_all_pose = self.fc_all_pose(comm_features.contiguous().view(comm_features.size(0), -1))
        return d_all_pose


class Discriminator(nn.Module):
    def __init__(self,
                num_joints=21,
                hidden_poseeach=32, 
                hidden_poseall=1024,
                poseall_num_layers=2,
                hidden_comm=32):
        super(Discriminator, self).__init__() 
        self.num_joints = num_joints
        self.hidden_comm = hidden_comm
        self.comm_conv = nn.Sequential(
            nn.Conv2d(9, self.hidden_comm, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.hidden_comm, self.hidden_comm, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.LeakyReLU(0.2),
            )
        
        self.disc_poseeach = PoseEachDiscriminator(num_joints=self.num_joints, hidden_poseeach=hidden_poseeach)
        self.disc_poseall = PoseAllDiscriminator(num_joints=self.num_joints, hidden_poseall=hidden_poseall, num_layers=poseall_num_layers)
    
    def forward(self, pose, input_type='matrot'):
        # input has a shape of SMPL pose, N x num_joints x 9
        if input_type == 'aa':
            pose = batch_rodrigues(pose.contiguous().view(-1,3))
            pose = pose.view(-1, self.num_joints, 9)
        inputs = pose.transpose(1, 2).unsqueeze(2) # to N x 9 x 1 x num_joints
        comm_features = self.comm_conv(inputs).view(-1, self.hidden_comm, self.num_joints) # to N x hidden_comm x num_joints
        d_poseeach = self.disc_poseeach(comm_features) # B x num_joints
        d_poseall = self.disc_poseall(comm_features) # B x 1
        d_out = {
            'poseeach' : d_poseeach,
            'poseall' : d_poseall,
        }
        return d_out
