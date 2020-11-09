"""Ponder Enhancer

This file implements the speech enhancement network. It is composed of an encoder, a request and scoring module (consists of a ponder mechanism supplemented by attention), and a decoder.
It imports the Attention Channel Ponder class from the ponder_utils module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ponder_utils import AttentionChannelPonder

def make_convs(params, two_d_convs):
    layers = []
    for i in range(params['n_layers']):
        if two_d_convs:
            layers.append(nn.Conv2d(params['in_channels'][i],
                                    params['out_channels'][i],
                                    params['kernel_size'][i],
                                    stride=params['stride'][i],
                                    padding=params['padding'][i],
                                    dilation=params['dilation'][i],
                                    bias=False))

            layers.append(nn.ReLU())
            layers.append((params['out_channels'][i]))

        else:
            layers.append(nn.Conv1d(params['in_channels'][i],
                                    params['out_channels'][i],
                                    params['kernel_size'][i],
                                    stride=params['stride'][i],
                                    padding=params['padding'][i],
                                    dilation=params['dilation'][i],
                                    bias=False))

            layers.append(nn.ReLU())
            layers.append(GlobalLayerNorm(params['out_channels'][i]))

    return nn.Sequential(*layers)

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)
    Taken from https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
    """
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(
            1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Parameters:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(
            dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1,
                                          keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / \
            torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y

class PonderEnhancer(nn.Module):
    """Ponder Enhancer Network
    This file implements the speech enhancment network. 
    It contains functions implemented the encoder, Attention Channel Ponder, and the decoder.
    """
    def __init__(self, params):
        """
        Parameters:
            params: Contains information regarding the following attributes
                stft_win_size: Window length of STFT
                stft_hop: Hop size of STFT
                hidden_size: Hidden size
                complex_mask: Boolean, if True uses real and imaginary masks to construct speech estimate
                two_d_convs: Boolean, if True model uses 2-D Convs, if False model uses 1-D Convs
                encoder_params: Dictionary of parameters for Convs in encoder, passed to function make_convs()
                decoder_params: Dictionary of parameters for Convs in decoder, passed to function make_convs()
        """
        super(PonderEnhancer, self).__init__()
        self.hidden_size = params['hidden_size']
        self.stft_win_size = params['stft_win_size']
        self.stft_size = self.stft_win_size // 2 + 1
        self.stft_hop = params['stft_hop']

        self.complex_mask = params['complex_mask']
        self.two_d_convs = params['two_d_convs']

        self.ponder_fn = AttentionChannelPonder(params)


        if self.complex_mask:
            self.mask_net = nn.Sequential(nn.Conv1d(self.stft_size,
                                                    2 * self.stft_size,
                                                    1, stride=1),
                                        nn.ReLU())
        else:
            self.mask_net = nn.Sequential(nn.Conv1d(self.stft_size,
                                                    self.stft_size,
                                                    1, stride=1),
                                        nn.Sigmoid())


        self.encode_convs = make_convs(params['encoder_params'], self.two_d_convs)
        self.decode_convs = make_convs(params['decoder_params'], self.two_d_convs)

    def encode(self, x):
        # batch x channels x time -> batch*channels x time
        x_reshape = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))

        # batch*channels x time -> batch*channels x nfilt x time x 2
        hann_window = torch.hann_window(window_length=self.stft_win_size,
                                        periodic=True,
                                        device=x.device)

        x_complex = torch.stft(x_reshape,
                            hop_length=self.stft_hop,
                            win_length=self.stft_win_size,
                            n_fft=self.stft_win_size,
                            window=hann_window,
                            onesided=True,
                            normalized=True)

        x_complex = x_complex.reshape((x.shape[0],
                                        x.shape[1],
                                        x_complex.shape[1],
                                        x_complex.shape[2],
                                        x_complex.shape[3]))

        # input will be batch x channels x time x input_dim
        x_mag, x_phase = torchaudio.functional.magphase(x_complex)

        # batch x channels x nfilt x time -> batch*channels x nfilt x time
        x_mag_reshape = x_mag.reshape(x_mag.shape[0]*x_mag.shape[1],
                                      x_mag.shape[2], x_mag.shape[3])

        if self.two_d_convs:
             x_mag_reshape = x_mag_reshape.unsqueeze(1)

        # CNN - extract features/downsample
        encoding_out = self.encode_convs(x_mag_reshape)

        if self.two_d_convs:
            encoding_out = encoding_out[:,0]

        encoding_out = encoding_out.reshape(x_mag.shape[0], x_mag.shape[1],
                                      encoding_out.shape[1], encoding_out.shape[2])

        return x_mag, x_phase, encoding_out

    def decode(self, ponder_out, ponder_costs, x_mag, x_phase):
        # Transpose CNN - upsampling
        if self.two_d_convs:
            ponder_out = ponder_out.unsqueeze(1)

        deconv_out = self.decode_convs(ponder_out)

        if self.two_d_convs:
            deconv_out = deconv_out[:,0]

        mask = self.mask_net(deconv_out)

        if self.complex_mask:
            masked_real = x_mag[:,0] * torch.cos(x_phase[:,0]) * mask[:,:self.stft_size]
            masked_imag = x_mag[:,0] * torch.sin(x_phase[:,0]) * mask[:,self.stft_size:]
            complex = torch.stack((masked_real, masked_imag), dim=-1)
        else:
            masked_mag = (x_mag[:,0,:,:] * mask)

            complex = torch.stack((masked_mag * torch.cos(x_phase[:,0]),
                                   masked_mag * torch.sin(x_phase[:,0])),
                                   dim=-1)

        hann_window = torch.hann_window(window_length=self.stft_win_size,
                                        periodic=True,
                                        device=mask.device)

        final = torch.istft(complex,
                            hop_length=self.stft_hop,
                            win_length=self.stft_win_size,
                            n_fft=self.stft_win_size,
                            window=hann_window,
                            onesided=True,
                            normalized=True)
        return final

    def forward(self, x, verbose=False):
        x_mag, x_phase, encoding_out = self.encode(x, verbose=verbose)

        ponder_out, ponder_costs = self.ponder_fn(encoding_out.transpose(-1, -2),
                                                    verbose=verbose)
        ponder_out = ponder_out.transpose(-1, -2)

        final = self.decode(ponder_out, ponder_costs, x_mag,
                            x_phase, verbose=verbose)

        return final, ponder_costs.mean()
