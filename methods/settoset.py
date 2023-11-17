# This code is modified from protonet.py

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class SetToSetBase(MetaTemplate):
    def __init__(self, backbone, n_way, n_support, score):
        super(SetToSetBase, self).__init__(backbone, n_way, n_support)
        self.score = score
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()

        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(
            1
        )  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Perform set-to-set transform
        z_proto = self.transform(z_proto)

        scores = self.score(z_query, z_proto)

        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores = self.set_forward(x)

        # Contrastive learning
        if self.training:
            z_support, z_query = self.parse_feature(x, False)
            num_data = (self.n_support + self.n_query) * self.n_way
            cl_task = torch.cat([z_support, z_query], 1)
            cl_task = cl_task.contiguous().view(
                -1, self.n_support + self.n_query, self.feat_dim
            )

            cl_embed = self.transform(cl_task)
            cl_embed = cl_embed.view(
                self.n_way, self.n_support + self.n_query, self.feat_dim
            )

            centers = (
                torch.mean(cl_embed, 1).contiguous().view(self.n_way, self.feat_dim)
            )
            cl_task = cl_task.view(num_data, self.feat_dim)

            cl_scores = self.score(cl_task, centers)

            cl_y_query = torch.from_numpy(
                np.repeat(range(self.n_way), self.n_support + self.n_query)
            )
            return self.loss_fn(scores, y_query) + self.loss_fn(cl_scores, cl_y_query)

        return self.loss_fn(scores, y_query)


class FEAT(SetToSetBase):
    def __init__(self, backbone, n_way, n_support, score):
        super(FEAT, self).__init__(backbone, n_way, n_support, score)
        self.transform = torch.nn.TransformerEncoderLayer(self.feat_dim, 1)


class EuclideanDistanceScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        distances = torch.pow(x - y, 2).sum(2)
        return -distances
