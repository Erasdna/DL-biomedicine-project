import torch
import torch.nn as nn
from torch.autograd import Variable

from methods.settoset.settoset import SetToSetBase

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first=True)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        c0 = Variable(torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        h0 = Variable(torch.rand(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        if torch.cuda.is_available():
            c0 = c0.cuda()
            h0 = h0.cuda()
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out + x
        return out.squeeze()

class FEALSTM(SetToSetBase):
    def __init__(self, backbone, n_way, n_support, score, num_layers):
        super(FEALSTM, self).__init__(backbone, n_way, n_support, score, None)
        self.transform = BidirectionalLSTM(self.feat_dim, num_layers, self.feat_dim // 2)

