import torch
import torch.nn as nn

class dummyloss(nn.Module):
    def __init__(self):
        super(dummyloss, self).__init__()
        pass


class DiscrLoss(nn.Module):
    def __init__(self,
            shape_w=1,
            poseeach_w=23,
            poseall_w=1,
            poseshape_w=1,
            reduction='mean',
            loss_type='L2'):
        super(DiscrLoss, self).__init__()

        self.discr_types = ['shape', 'poseeach', 'poseall', 'poseshape']
        self.discr_weights = dict(zip(self.discr_types, 
                                        [shape_w, poseeach_w, poseall_w, poseshape_w]))
        self.shape_w = shape_w
        self.poseeach_w = poseeach_w
        self.poseall_w = poseall_w
        self.poseshape_w = poseshape_w

        self.reduction = reduction
        self.loss_type = loss_type

    def compute_loss_per_d(self, d, label):
        '''
        d - tensor of size B x N
        label can be any real value (typically 0 or 1)
        
        Discriminators are trained in regression fashion
        '''
        if self.loss_type=='L2':
            out = (d - label) ** 2
        elif self.loss_type == 'L1':
            out = torch.abs(d - label)
        else:
            raise NotImplementedError

        return out / out.size(1) # divide by the size of the discr output

    def forward(self, d, label):
        '''
        d - dict with discr outs for shape, poseeach, poseall, poseshape
        '''
        res = []
        d_names = []
        for discr_type in self.discr_types:
            if discr_type in d:
                out = self.compute_loss_per_d(d[discr_type], label)
                out = out * self.discr_weights[discr_type]
                res.append(out)
                d_names.append(discr_type)

        if self.reduction == 'mean':
            res = torch.cat(res, dim=1)
            return res.mean()
        # elif self.reduction == None:
        #     return dict(zip(d_names, res))
        else:
            raise NotImplementedError


