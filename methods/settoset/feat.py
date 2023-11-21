import torch

from methods.settoset.settoset import SetToSetBase

class FEAT(SetToSetBase):
    """
    Implementation of the FEAT model.
    Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/models/feat.py and https://arxiv.org/abs/1812.03664.
    """    
    def __init__(self, backbone, n_way, n_support, score, n_head, dim_feedforward, dropout):
        """Constructor of FEAT class.

        Args:
            backbone (nn.Module): Backbone for providing embeddings.
            n_way (int): number of classes.
            n_support (int): number of samples in support set.
            score (nn.Module): Module for evaluating score based on the embeddings.
            n_head (int): number of heads in the transformer layer.
            dim_feedforward (int): size of the feedforward layer in the transformer layer.
            dropout (float): dropout in the transformer layer.
        """
        super(FEAT, self).__init__(backbone, n_way, n_support, score, None)
        self.transform = torch.nn.TransformerEncoderLayer(self.feat_dim, n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        
