import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num):
        super(Network, self).__init__()
        self.resnet = resnet
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        # self.instance_projector = nn.Sequential(
        #     nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.resnet.rep_dim, self.feature_dim),
        # )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, 512),  # self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(512, self.cluster_num),
            nn.Softmax(dim=1)
        )

        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
        #     nn.Tanh()
        # )

    def forward(self, x_i, x_j):
        h_i = self.resnet(x_i)
        h_j = self.resnet(x_j)

        # z_i = normalize(self.instance_projector(h_i), dim=1)
        # z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return h_i, h_j, c_i, c_j  # z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c

    def forward_feature_map(self, x):
        """
        :param x:
        :return: output of backbone, instance head, and cluster head
        """
        out_backbone = self.resnet(x)
        # out_instance = self.instance_projector(out_backbone)
        out_cluster = self.cluster_projector(out_backbone)
        return out_backbone, out_cluster
