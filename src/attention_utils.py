"""Encoder Layer 

This file contains helper functions to implement an encoder layer.
This implementation is based on 'Attention is All You Need', Vaswani et al., 2017
See Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
          and https://github.com/bentrevett/pytorch-seq2seq
"""

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionEncoderLayer(nn.Module):
    """Attention Encoder Layer 
    Represents one Encoder layer, used to aid in aggregation across channels to preoduce an internal representation which is assined a score.
    """
    def __init__(self, input_size, hidden_size, n_heads, dropout):
        """
        Parameters:
            input_size: Window length of STFT
            hidden_size: Hidden Size
            n_heads: Number of attention heads
            dropout: Dropout probability used after attention and feed-forward network
        """
        super(AttentionEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, n_heads, dropout)

        self.feedforward_norm = nn.LayerNorm(hidden_size)
        self.positionwise_feedforward = PositionwiseFeedforward(hidden_size,
                                        hidden_size * 2,
                                        dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, verbose=False):
        # batchsize x num_frames x hidden -> num_frames x batchsize x query len x hidden
        # query len is always 1 in these experiments
        query = query.reshape(query.shape[1], query.shape[0], -1, query.shape[2])
        key = key.reshape(key.shape[2], key.shape[0], key.shape[1], key.shape[3])
        value = value.reshape(value.shape[2], value.shape[0], value.shape[1], value.shape[3])

        att_out, attention = self.attention(query, key, value, verbose=verbose)
        att_dropout = self.dropout(att_out)
        query = self.attention_norm(query + att_dropout)

        feedfwd_out = self.positionwise_feedforward(query)
        feedfwd_dropout = self.dropout(feedfwd_out)
        query = self.feedforward_norm(query + feedfwd_dropout)
        query = query.reshape(query.shape[1], query.shape[0], query.shape[3])

        return query

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanism
    Represents first componenet implemented as part of the Encoder layer.
    """
    def __init__(self, hidden_size, n_heads, dropout):
        """
        Parameters:
            hidden_size: Hidden size
            n_heads: Number of attention heads
            dropout: Dropout probability used after applying softmax to attention vector
        """
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % n_heads == 0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = hidden_size//n_heads

        self.linear_Q = nn.Linear(hidden_size, hidden_size)
        self.linear_K = nn.Linear(hidden_size, hidden_size)
        self.linear_V = nn.Linear(hidden_size, hidden_size)
        self.linear_Wnot = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_size]))

    def forward(self, query, key, value):
        batch_size = query.shape[1]
        num_frames = query.shape[0]

        Q = self.linear_Q(query)
        K = self.linear_K(key)
        V = self.linear_V(value)

        Q = Q.view(num_frames, batch_size, -1, self.n_heads, self.head_size).permute(0, 1, 3, 2, 4)
        K = K.view(num_frames, batch_size, -1, self.n_heads, self.head_size).permute(0, 1, 3, 2, 4)
        V = V.view(num_frames, batch_size, -1, self.n_heads, self.head_size).permute(0, 1, 3, 2, 4)

        energy = torch.matmul(Q, K.permute(0, 1, 2, 4, 3))/ self.scale.to(query.device)

        attention = torch.softmax(energy, dim = -1)
        
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(num_frames, batch_size, -1, self.hidden_size)

        x = self.linear_Wnot(x)

        return x, attention

class PositionwiseFeedforward(nn.Module):
    """Position-wse FeedForward Network
    Represents second componenet implemented as part of the Encoder layer.
    """
    def __init__(self, input_dim, output_dim, dropout):
        """
        Parameters:
            input_dim: Input dimension to FFN (set to Hidden Size) 
            output_dim: Output dimension of FFN (set to Hidden Size * 2)
            dropout:  Dropout probability used after ReLU
        """
        super(PositionwiseFeedforward, self).__init__()
        self.fc_1 = nn.Linear(input_dim, output_dim)
        self.fc_2 = nn.Linear(output_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x
