from torch import nn
import torch.nn.functional as F
import torch

class CrossCorrelationLoss(nn.Module):

    def __init__(self, n_class, lambd, device):
        super(CrossCorrelationLoss, self).__init__()
        self.n_class = n_class
        self.lambd = lambd
        self.device = device
        self.bn = nn.BatchNorm1d(n_class, affine=False)

    def forward(self, y_i, y_j):
        batch_size = y_i.size(0)
        c = self.bn(y_i).T @ self.bn(y_j)
        # sum the cross-correlation matrix between all gpus
        c.div_(batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
