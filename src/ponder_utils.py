"""Attention Channel Ponder

This file contains helper functions to implement Data Request Mechanism, where the model iteratively requests to stream from microphones based on the results obtained from the Attention Encoder Layer. 
It imports the Attention Encoder Layer from the attention_utils module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pdb
from attention_utils import AttentionEncoderLayer

class AttentionChannelPonder(nn.Module):
    """Attention Channel Ponder Mechanism
    Scoring and request module that determines if model should request to stream more data from another microphone. 
    Also provides interface to run baseline experiments.
    """
    def __init__(self, params):
        """
        Parameters:
            params: Contains information regarding the following attributes
                stft_win_size: Window length of STFT
                hidden_size: Hidden size
                n_heads: Number of attention heads
                dropout: Dropout probability to be used primarily in Attention Encoder Layer
                fixed_k: If running baseline experiments, fixed number of microphones to request data from
        """
        super(AttentionChannelPonder, self).__init__()
        self.stft_win_size = params['stft_win_size']
        self.stft_size = self.stft_win_size // 2 + 1
        self.hidden_size = params['hidden_size']
        self.n_heads = params['n_heads']
        self.dropout = params['dropout']


        self.fixed_k = None
        if 'fixed_k' in params:
            self.fixed_k = params['fixed_k']

        # denoising layers
        self.dense_in = nn.Sequential(nn.Linear(self.stft_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Dropout(p=self.dropout))

        self.dense_out = nn.Sequential(nn.Linear(self.hidden_size, self.stft_size),
                                        nn.ReLU(),
                                        nn.Dropout(p=self.dropout))

        self.attention_encoder = AttentionEncoderLayer(self.stft_win_size,
                                                        self.hidden_size,
                                                        self.n_heads,
                                                        self.dropout)

        # attention over h
        self.seen_enough = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2),
                                         nn.ReLU(),
                                         nn.Dropout(p=0.15),
                                         nn.Linear(self.hidden_size//2, 1),
                                         PSigmoid())

    def forward(self, x):
        lin_1_out = self.dense_in(x)
        encoder_out = torch.Tensor(lin_1_out.size(0), lin_1_out.size(2), self.hidden_size).to(x.device)
        all_ponder_costs = torch.zeros(encoder_out.size(1)).to(x.device)

        eps = 1e-4
        still_pondering = torch.ones(x.shape[0], x.shape[2], 1).to(x.device)
        cum_halt_score = torch.zeros(x.shape[0], x.shape[2], 1).to(x.device)
        cur_halt_score = torch.zeros(x.shape[0], x.shape[2], 1).to(x.device)
        remainder_value = torch.ones(x.shape[0], x.shape[2], 1).to(x.device)
        ponder_cost = torch.zeros(x.shape[0], x.shape[2], 1).to(x.device)

        h = lin_1_out[:, 0, :, :]
        output = torch.zeros_like(h).float()

        if self.fixed_k is not None:
            ponder_cost += self.fixed_k
            query = lin_1_out[:, :self.fixed_k].mean(1)
            key = lin_1_out[:, :self.fixed_k]
            value = lin_1_out[:, :self.fixed_k]

            output = self.attention_encoder(query, key, value, verbose=False)

        else:
            for ch_idx in range(lin_1_out.size(1)):
                query = lin_1_out[:,ch_idx]
                key = lin_1_out[:,:ch_idx + 1]
                value = lin_1_out[:,:ch_idx + 1]

                h = self.attention_encoder(query, key, value, verbose=False)

                if ch_idx < lin_1_out.size(1)-1:
                    cur_halt_score = self.seen_enough(h)
                else:
                    cur_halt_score = torch.ones_like(cur_halt_score)

                cum_halt_score = cum_halt_score + cur_halt_score * still_pondering
                ponder_cost = ponder_cost + still_pondering

                still_pondering = (cum_halt_score < (1 - eps)).float()
                output = output + h * (cur_halt_score * still_pondering + remainder_value * (1 - still_pondering))
                remainder_value = remainder_value - cur_halt_score * still_pondering
                ponder_cost = ponder_cost + remainder_value * (1 - still_pondering)
                
                if still_pondering.sum() < eps:
                    break

        all_ponder_costs = ponder_cost[:,:,0]
        encoder_out = self.dense_out(output)

        return encoder_out, all_ponder_costs

class PSigmoid(nn.Module):
    def __init__(self, a_init=0.0, b_init=1.0):
        super(PSigmoid, self).__init__()
        self.a = nn.Parameter(torch.Tensor([a_init]))
        self.b = nn.Parameter(torch.Tensor([b_init]))
        self.sig = nn.Sigmoid()
        self.eps = 1e-8
    def forward(self, x):
        return self.sig((x - self.a) / (self.b + self.eps))
