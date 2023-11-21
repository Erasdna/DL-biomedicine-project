# This code is based on protonet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from methods.meta_template import MetaTemplate


class SetToSetBase(MetaTemplate):
    """
    Base class for set-to-set transformation models based on paper "Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions" (https://arxiv.org/abs/1812.03664).
    Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/models/feat.py and https://arxiv.org/abs/1812.03664.
    """

    def __init__(self, backbone, n_way, n_support, score, transform):
        """Constructor of SetToSetBase class.

        Args:
            backbone (nn.Module): Backbone for providing embeddings.
            n_way (int): number of classes.
            n_support (int): number of samples in support set.
            score (nn.Module): Module for evaluating score based on the embeddings.
            transform (nn.Module): Module to be used as the set-to-set transformation function.
        """
        super(SetToSetBase, self).__init__(backbone, n_way, n_support)
        self.score = score
        self.transform = transform
        self.loss_fn = nn.CrossEntropyLoss()

    def set_forward(self, x, is_feature=False):
        # Parse features to get support and query embeddings
        z_support, z_query = self.parse_feature(x, is_feature)

        z_support = z_support.contiguous()

        z_proto = z_support.view(self.n_way, self.n_support, -1).mean(
            1
        )  # the shape of z is [n_data, n_dim]
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        # Perform set-to-set transform
        z_proto = self.transform(z_proto)

        # Calculate score
        scores = self.score(z_query, z_proto)

        return scores

    def set_forward_loss(self, x):
        # Calculate scores of the input embeddings
        scores = self.set_forward(x)

        # Standard ProtoNet objective
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query)
        if torch.cuda.is_available():
            y_query = y_query.cuda()
        loss = self.loss_fn(scores, y_query)

        # Contrastive learning objective (only when training)
        if self.training:
            # Parse features
            z_support, z_query = self.parse_feature(x, False)

            num_data = (self.n_support + self.n_query) * self.n_way

            # Combine and reshape combined embeddings
            cl_task = torch.cat([z_support, z_query], 1)
            cl_task = cl_task.contiguous().view(
                -1, self.n_support + self.n_query, self.feat_dim
            )

            # Transform combined embeddings
            cl_embed = self.transform(cl_task)
            cl_embed = cl_embed.view(
                self.n_way, self.n_support + self.n_query, self.feat_dim
            )

            # Calculate new centers
            centers = (
                torch.mean(cl_embed, 1).contiguous().view(self.n_way, self.feat_dim)
            )
            cl_task = cl_task.view(num_data, self.feat_dim)

            # Calculate score
            cl_scores = self.score(cl_task, centers)

            # Create expected labels
            cl_y_query = torch.from_numpy(
                np.repeat(range(self.n_way), self.n_support + self.n_query)
            )
            cl_y_query = Variable(cl_y_query)
            if torch.cuda.is_available():
                cl_y_query = cl_y_query.cuda()

            # Calculate loss
            loss += self.loss_fn(cl_scores, cl_y_query)

        return loss


class EuclideanDistanceScore(nn.Module):
    """Euclidean distance-based scoring function. Returns the negative Euclidean distance as the score."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """Calculates score based on Euclidean distance of points.

        Args:
            x (torch.Tensor): tensor of N points with D dimensions, shape (N, D).
            y (torch.Tensor): tensor of M points with D dimensions, shape (M, D).

        Returns:
            torch.Tensor: tensor containing pair-wise scores, shape (N, M).
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        distances = torch.pow(x - y, 2).sum(2)
        return -distances


class CosineSimilarityScore(nn.Module):
    """Cosine similarity-based scoring function. Returns the cosine similarity as the score."""

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """Calculates score based on cosine similarity between the points.

        Args:
            x (torch.Tensor): tensor of N points with D dimensions, shape (N, D).
            y (torch.Tensor): tensor of M points with D dimensions, shape (M, D).

        Returns:
            torch.Tensor: tensor containing pair-wise scores, shape (N, M).
        """
        assert (
            x.shape[1] == y.shape[1]
        ), "Tensors must have the same number of dimensions D."
        score = F.normalize(x) @ F.normalize(y).T
        return score
