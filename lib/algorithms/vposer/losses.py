import torch
from torch import nn


class geodesic_loss_R(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(geodesic_loss_R, self).__init__()

        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self, m1, m2):
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, -1 + self.eps, 1 - self.eps)

        return torch.acos(cos)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred, ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        if self.reduction == 'batchmean':
            return torch.mean(torch.sum(theta, dim=theta.shape[1:]))
        else:
            return theta
