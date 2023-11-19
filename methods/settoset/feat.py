import torch

from methods.settoset.settoset import SetToSetBase

class FEAT(SetToSetBase):
    def __init__(self, backbone, n_way, n_support, score, n_head, dim_feedforward, dropout):
        super(FEAT, self).__init__(backbone, n_way, n_support, score, None)
        self.transform = torch.nn.TransformerEncoderLayer(self.feat_dim, n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        