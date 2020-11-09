""" 
Loss function selector for training a model.
"""

import torch
import numpy as np

def get_loss_fn(loss_config):
    def sdr_loss(y, y_hat):
            return -10 * torch.log10((y_hat * y_hat).mean(-1) \
                    / ((y_hat - y)**2).mean(-1)).mean()

    def power_mag_stft_loss(y, y_hat, win_size=512, hop=128, p=.3):
        hann_window = torch.hann_window(window_length=win_size,
                                        periodic=True,
                                        device=y.device)
        y_complex = torch.stft(y,
                            hop_length=hop,
                            win_length=win_size,
                            n_fft=win_size,
                            window=hann_window,
                            onesided=True,
                            normalized=True)

        y_mag = ((y_complex**2).sum(-1) + 1e-7)**p

        y_hat_complex = torch.stft(y_hat,
                            hop_length=hop,
                            win_length=win_size,
                            n_fft=win_size,
                            window=hann_window,
                            onesided=True,
                            normalized=True)

        y_hat_mag = ((y_hat_complex**2).sum(-1) + 1e-7)**p

        return ((y_mag - y_hat_mag)**2).mean()

    def power_mag_stft_loss_l1(y, y_hat, win_size=512, hop=128, p=.3):
        hann_window = torch.hann_window(window_length=win_size,
                                        periodic=True,
                                        device=y.device)
        y_complex = torch.stft(y,
                               hop_length=hop,
                               win_length=win_size,
                               n_fft=win_size,
                               window=hann_window,
                               onesided=True,
                               normalized=True)

        y_mag = ((y_complex**2).sum(-1) + 1e-7)**p

        y_hat_complex = torch.stft(y_hat,
                                   hop_length=hop,
                                   win_length=win_size,
                                   n_fft=win_size,
                                   window=hann_window,
                                   onesided=True,
                                   normalized=True)

        y_hat_mag = ((y_hat_complex**2).sum(-1) + 1e-7)**p

        return (torch.abs(y_mag - y_hat_mag)).mean()

    def mag_stft_loss(y, y_hat, win_size=512, hop=128):
        hann_window = torch.hann_window(window_length=win_size,
                                        periodic=True,
                                        device=y.device)
        y_complex = torch.stft(y,
                            hop_length=hop,
                            win_length=win_size,
                            n_fft=win_size,
                            window=hann_window,
                            onesided=True,
                            normalized=True)

        y_mag = (y_complex**2).sum(-1)

        y_hat_complex = torch.stft(y_hat,
                            hop_length=hop,
                            win_length=win_size,
                            n_fft=win_size,
                            window=hann_window,
                            onesided=True,
                            normalized=True)

        y_hat_mag = (y_hat_complex**2).sum(-1)

        return ((y_mag - y_hat_mag)**2).mean()

    if loss_config['enhance_loss_fn'] == 'sdr':
        enhance_loss_fn = sdr_loss
    elif loss_config['enhance_loss_fn'] == 'mag_stft':
        enhance_loss_fn = mag_stft_loss
    elif loss_config['enhance_loss_fn'] == 'power_mag_stft':
        enhance_loss_fn = power_mag_stft_loss
    elif loss_config['enhance_loss_fn'] == 'power_mag_stft_l1':
        enhance_loss_fn = power_mag_stft_loss_l1

    def ponder_loss(ponder):
         return ponder.mean()

    ponder_loss_fn = ponder_loss

    def total_loss(y, y_hat, ponder):
        return enhance_loss_fn(y, y_hat), ponder_loss_fn(ponder)

    return total_loss
