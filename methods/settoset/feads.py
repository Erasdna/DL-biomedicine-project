import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.settoset.settoset import SetToSetBase


class DeepSetsFunc(nn.Module):
    def __init__(self, z_dim):
        super(DeepSetsFunc, self).__init__()
        """
        DeepSets transformation function.
        Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/models/deepset.py
        """
        self.gen1 = nn.Linear(z_dim, z_dim * 4)
        self.gen2 = nn.Linear(z_dim * 4, z_dim)
        self.gen3 = nn.Linear(z_dim * 2, z_dim * 4)
        self.gen4 = nn.Linear(z_dim * 4, z_dim)
        self.z_dim = z_dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input.

        Returns:
            torch.Tensor: DeepSets output
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        _, L, D = x.shape
        assert D == self.z_dim

        # Ones mask
        mask_one = torch.ones(L, L) - torch.eye(L, L)
        mask_one = mask_one.view(1, L, L, 1)
        if torch.cuda.is_available():
            mask_one = mask_one.cuda()

        combined_mean = torch.mul(x.unsqueeze(2), mask_one).max(1)[0]
        # Bilinear transform
        combined_mean = F.relu(self.gen1(combined_mean.view(-1, self.z_dim)))
        combined_mean = self.gen2(combined_mean)
        combined_mean_cat = torch.cat(
            [x.contiguous().view(-1, self.z_dim), combined_mean], 1
        )
        # Linear transform
        combined_mean_cat = F.relu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        combined_mean_cat = combined_mean_cat.view(-1, L, self.z_dim)

        # Residual layer
        set_output = x + combined_mean_cat

        x = set_output.squeeze()
        return x


class FEADS(SetToSetBase):
    """
    Implementation of the DeepSets variant of the set-to-set transformation model.
    Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/models/deepset.py
    """
    def __init__(self, backbone, n_way, n_support, score):
        """Constructor of FEADS (DeepSets transformation) class.

        Args:
            backbone (nn.Module): Backbone for providing embeddings.
            n_way (int): number of classes.
            n_support (int): number of samples in support set.
            score (nn.Module): Module for evaluating score based on the embeddings.
        """
        super(FEADS, self).__init__(backbone, n_way, n_support, score, None)
        self.transform = DeepSetsFunc(self.feat_dim)
