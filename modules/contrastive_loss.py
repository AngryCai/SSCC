import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z)

        sim = torch.matmul(z, z.T) / self.temperature  # Dot similarity
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


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


class ConsistencyLoss(nn.Module):

    def __init__(self, device):
        super(ConsistencyLoss, self).__init__()
        self.device = device

    def forward(self, y_i, y_j):
        consistency_loss = F.mse_loss(y_i, y_j)
        return consistency_loss


class PseudoLabelBinaryCE(nn.Module):
    def __init__(self, device, threshold=0.9):
        super(PseudoLabelBinaryCE, self).__init__()
        self.device = device
        self.threshold = threshold
        # self.cosine_sim = nn.CosineSimilarity(eps=1e-6).to(self.device)
        self.bce_criterion = nn.BCELoss(reduction='mean').to(self.device)

    # def pair_wise_simlarity(self, x, type='cos'):
    #     assert type != 'cos' or type != 'inner', print('similarity type error!')
    #     batch_size, n_fea = x.size(0), x.size(1)
    #     if type == 'cos':
    #         x1 = x.repeat((batch_size, 1)).to(self.device)
    #         x2 = x.repeat((1, batch_size)).reshape((-1, n_fea)).to(self.device)
    #         return self.cosine_sim(x1, x2) #.reshape((batch_size, -1))
    #     else:
    #         return torch.mm(x, x.T).reshape(-1)

    def forward(self, h_i, h_j, y_i, y_j):
        h = torch.cat((h_i, h_j), dim=0)
        y = torch.cat((y_i, y_j), dim=0)
        h = F.normalize(h)
        cos_sim = torch.matmul(h, h.T).reshape(-1)  # cosine similarity
        # P_ = self.pair_wise_simlarity(h, type='cos')  # 1: same class, -1: different class, 0: not sure
        labels = torch.zeros_like(cos_sim).float().to(self.device)
        labels[cos_sim >= self.threshold] = 1
        labels[cos_sim <= 0] = -1
        logits = torch.matmul(y, y.T).reshape(-1)
        if labels.max().item() == 0.0:
            print(labels.max().item(), labels.min().item())
        # self.pair_wise_simlarity(y, type='inner')
        return self.bce_criterion(logits, labels)


class CenterContrastiveLoss(nn.Module):
    def __init__(self, num_class, temperature, device):
        super(CenterContrastiveLoss, self).__init__()
        self.device = device
        self.num_class = num_class
        self.temperature = temperature
        self.contrastive_loss = InstanceLoss(self.num_class, self.temperature, self.device)

    def forward(self, h_i, h_j, y_i, y_j):
        c_i = (1. / torch.sum(y_i, 0).unsqueeze(1)) * torch.matmul(y_i.T, h_i)
        c_j = (1. / torch.sum(y_j, 0).unsqueeze(1)) * torch.matmul(y_j.T, h_j)
        return self.contrastive_loss(c_i, c_j)


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
