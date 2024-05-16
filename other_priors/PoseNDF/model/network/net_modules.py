import torch
import torch.nn as nn
from model.network.net_utils import get_parent_mapping



class DFNet(nn.Module):

    def __init__(self, opt, batch_size=4, use_gpu=0, layer='UpperClothes', weight_norm=True, activation='relu',
                dropout=0.3, output_layer=None):
        super().__init__()
        input_size = opt['in_dim']
        hid_layer = opt['dims'].split(',')
        hid_layer = [int(val) for val in hid_layer]
        output_size = 1
        dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            # if weight_norm:
            #     lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        if opt['act'] == 'lrelu':    
            self.actv = nn.LeakyReLU()
            self.out_actv = nn.ReLU()
        
        if opt['act'] == 'relu':    
            self.actv = nn.ReLU()
            self.out_actv = nn.ReLU()
        

        if opt['act'] == 'softplus':    
            self.actv = nn.Softplus(beta=opt['beta'])
            self.out_actv = nn.Softplus(beta=opt['beta'])
        



    def forward(self, p):


        x = p.reshape(len(p), -1)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
  
            if l < self.num_layers - 2:
                x =  self.actv(x)

        x =  self.out_actv(x)

        return x


class BoneMLP(nn.Module):
    """from LEAP code(CVPR21, Marko et al)"""

    def __init__(self, bone_dim, bone_feature_dim,parent=-1, act='relu', beta=100.):
        super(BoneMLP, self).__init__()
        if parent ==-1:
            in_features = bone_dim
        else:
            in_features = bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        if act  =='relu':
            self.net = nn.Sequential(
                nn.Linear(in_features, n_features),
                nn.ReLU(),
                nn.Linear(n_features, bone_feature_dim),
                nn.ReLU()
            )

        if act  =='lrelu':
            self.net = nn.Sequential(
                nn.Linear(in_features, n_features),
                nn.LeakyReLU(),
                nn.Linear(n_features, bone_feature_dim),
                nn.LeakyReLU()
            )
        if act  =='softplus':
            self.net = nn.Sequential(
                nn.Linear(in_features, n_features),
                nn.Softplus(beta=beta),
                nn.Linear(n_features, bone_feature_dim),
                nn.Softplus(beta=beta)
            )

    def forward(self, bone_feat):

        return self.net(bone_feat)


class StructureEncoder(nn.Module):
    """from LEAP code(CVPR21, Marko et al)"""
    def __init__(self,opt, local_feature_size=6):
        super().__init__()

        self.bone_dim = 4  # 3x3 for pose and 1x3 for joint loc  #todo: change this encodibg for quaternion
        self.input_dim = self.bone_dim  # +1 for bone length
        self.parent_mapping = get_parent_mapping('smpl')

        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size



        self.net = nn.ModuleList([ BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i], opt['act'], opt['beta']) for i in range(self.num_joints) ])

    def get_out_dim(self):
        return self.out_dim

    @classmethod
    def from_cfg(cls, config):
        return cls(
            local_feature_size=config['local_feature_size'],
            parent_mapping=config['parent_mapping']
        )

    def forward(self, quat):
        """
        Args:
            pose: B x num_joints x 4
            rel_joints: B x num_joints x 3
        """
        B = quat.shape[0]
        # quat = quat.reshape(B, 21, 4)

        # B, K = rel_joints.shape[0], rel_joints.shape[1]
        # bone_lengths = torch.norm(rel_joints.squeeze(-1), dim=-1).view(B, K, 1)  # B x num_joints x 1
        #
        # bone_features = torch.cat((pose.contiguous().view(B, K, -1),
        #                            rel_joints.contiguous().view(B, K, -1)), dim=-1)
        #
        # root_bone_prior = self.proj_bone_prior(bone_features.contiguous().view(B, -1))  # B, bottleneck
        # root_bone_prior = self.proj_bone_prior(quat.contiguous().view(B, -1))  # B, bottleneck

        # fwd pass through the bone encoder
        features = [None] * self.num_joints
        # bone_transforms = torch.cat((bone_features, bone_lengths), dim=-1)
        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(quat[:, i, :])
            else:
                inp = torch.cat((quat[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
        features = torch.cat(features, dim=-1)  # B x f_len
        return features