"""
Anubhav Natani
2019
Pytorch
Experimentation model based upon the torch library
Easy for hyper parameter tuning and embedding setting and other such stuff
Refrences -
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IS_CUDA = torch.cuda.is_available()
if(IS_CUDA):
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Encoder
# For Sequence to Sequence Model
# sequence of layers in encoder
"""
1.Input
2.Embedding Layer
3.Gru(prev input)
4.Output and hidden
"""


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0):
        # prev set layers = 1 and dropout is zero
        super(Encoder, self).__init__()
        # initialization
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        # embedding layer creation
        """
        Self embedding can be tried
        """
        #self.embedding = embedding

        self.embedding = nn.Embedding(input_size, hidden_size)
        # Biderictional Gated Recurrent unit defination
        """
        Modification Chance
        """
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=self.dropout, bidirectional=True)

        # forward method of the model
    def forward(self, input_data, input_len, hidden=None):
        # multiple batches run on a single iteration
        embed = self.embedding(input_data)
        # packed sequence for rnn creation
        # pack padded sequence of variable length
        packed = torch.nn.utils.rnn.pack_padded_sequence(embed, input_len)
        # packing done for the rnn input
        # previous hidden also put in gru
        out, hidden = self.gru(packed, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out)
        # bidirectional sum
        out = out[:, :, :self.hidden_size]+out[:, :, self.hidden_size]
        # to be used in the next pass
        return out, hidden

# Attention Layer
