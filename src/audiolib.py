# -*- coding: utf-8 -*-
"""
MODIFIED FROM https://github.com/microsoft/MS-SNSD
Created on Wed Jun 26 15:54:05 2019

@author: chkarada
"""
import soundfile as sf
import os
import numpy as np

# Function to read audio
def audioread(path, norm=True, start=0, stop=None):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        x, sr = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')

    if len(x.shape) == 1:  # mono
        if norm:
            rms = (x ** 2).mean() ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar
        return x, sr
    else:  # multi-channel
        x = x.T
        if norm:
            rms = (x ** 2).mean(1) ** 0.5
            scalar = 10 ** (-25 / 20) / (rms)
            x = x * scalar[:,None]
        return x, sr

# Funtion to write audio
def audiowrite(data, fs, destpath, norm=False):
    if norm:
        data = data / np.max(np.abs(data), axis=0)[None,]

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, data, fs)

# Function to mix clean speech and noise at various SNR levels
def snr_setter(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5

    rmsnoise = (noise**2).mean()**0.5
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5

    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    return clean, noisenewlevel
