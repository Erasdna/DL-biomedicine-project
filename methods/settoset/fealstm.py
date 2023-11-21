import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.settoset.settoset import SetToSetBase


class BidirectionalLSTM(nn.Module):
    """
    Implementation of the Bidirectional LSTM.
    Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/models/bilstm.py.
    """

    def __init__(self, input_size, num_layers, hidden_size):
        """Constructor of the BidirectionalLSTM.

        Args:
            input_size (int): input embeddings dimension, in our case dimension of the backbone output embedding.
            num_layers (int): number of LSTM layers.
            hidden_size (int): hidden embedding dimension.
        """
        super().__init__()
        # Construct torch.nn.LSTM with given parameters
        self.lstm = nn.LSTM(
            input_size=input_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input.

        Returns:
            torch.Tensor: BiLSTM output
        """
        # Add batch dimension if it doesn't exist
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        # Initialize cell states
        c0 = Variable(
            torch.rand(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size),
            requires_grad=False,
        )
        # Initialize hidden states
        h0 = Variable(
            torch.rand(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size),
            requires_grad=False,
        )

        # Move to CUDA
        if torch.cuda.is_available():
            c0 = c0.cuda()
            h0 = h0.cuda()

        # Run through LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Residual connection
        out = out + x

        # Remove batch dimension if only 1 batch
        out = out.squeeze()

        return out


class FEALSTM(SetToSetBase):
    """
    Implementation of the Bidirectional LSTM variant of the set-to-set transformation model.
    Adapted from https://github.com/Sha-Lab/FEAT/blob/master/model/models/bilstm.py and https://arxiv.org/abs/1812.03664.
    """

    def __init__(self, backbone, n_way, n_support, score, num_layers):
        """Constructor of the class.

        Args:
            backbone (nn.Module): Backbone for providing embeddings.
            n_way (int): number of classes.
            n_support (int): number of samples in support set.
            score (nn.Module): Module for evaluating score based on the embeddings.
            num_layers (int): number of LSTM layers.
        """
        super(FEALSTM, self).__init__(backbone, n_way, n_support, score, None)
        self.transform = BidirectionalLSTM(
            self.feat_dim, num_layers, self.feat_dim // 2
        )
