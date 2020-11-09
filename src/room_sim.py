""" 
This file is called by data_gen.py using a config from data_gen_config.py to simulate a room. 
It uses the pyroomacoustics package and simulates movement by fading sounds in and out between different source locations.
We also include a simpler simulate_k_room method for testing model with no movement and where the 
right number of microphone to select can be modeled by a negative hypergeometric distribution. We used this to make sure
our selection mechanism performed as expected before attempting more complicated scenes.
"""

import numpy as np
from numpy.random import uniform as runif

import pyroomacoustics as pra
import audiolib


def fade_signal(signal, start_idx, end_idx, chunk_size, fade_overlap, num_locs, i):
    # fade in and out so we dont get any artifacts
    fade_in_mult = np.linspace(0, 1, fade_overlap)
    fade_out_mult = fade_in_mult[::-1]

    # if the start then no fade in only fade out
    if i == 0:
        signal[end_idx - len(fade_out_mult):end_idx] *= fade_out_mult

    # if the end then only fade in no fade out
    elif i == (num_locs - 1):
        signal[start_idx:start_idx + len(fade_in_mult)] *= fade_in_mult

    # ow fade in and fade out
    else:
        signal[start_idx:start_idx + len(fade_in_mult)] *= fade_in_mult
        signal[end_idx - len(fade_out_mult):end_idx] *= fade_out_mult

    return signal

def simulate_room(clean, noise, cfg, rand_seed, debug=False):
    n_mics = cfg['num_mics']
    n_inter = cfg['n_inter_locs']
    room_size_upper = cfg['room_size_upper']
    room_size_lower = cfg['room_size_lower']
    room_geom_fixed = cfg['room_geom_fixed']
    
    # each mixture has the same room across db levels
    np.random.seed(rand_seed)

    if room_geom_fixed:
        room_size = np.array([room_size_upper] * 3)

        # random seed holding so that we get the same random room but not the
        # same source source locs
        cur_rand_state = np.random.get_state()
        np.random.seed(42)
        mic_locs = [np.random.uniform(0, dim, n_mics) for dim in room_size]
        np.random.set_state(cur_rand_state)

    else:
        room_size = np.random.uniform(room_size_lower, room_size_upper, size=(3))
        mic_locs = [np.random.uniform(0, dim, n_mics) for dim in room_size]

    room = pra.ShoeBox(room_size,
                        fs=cfg['fs'],
                        absorption=0.35,
                        max_order=10)

    mic_locs = np.array(mic_locs).T
    mic_array = pra.MicrophoneArray(mic_locs.T, room.fs)
    room.add_microphone_array(mic_array)

    speech_locs = [np.linspace(runif(0, dim), runif(0, dim), n_inter) for i, dim in enumerate(room_size)]
    speech_locs = np.array(speech_locs).T

    noise_locs = [np.linspace(runif(0, dim), runif(0, dim), n_inter) for i, dim in enumerate(room_size)]
    noise_locs = np.array(noise_locs).T

    chunk_size = len(clean) // n_inter
    fade_overlap = chunk_size // 4
    for i in range(len(speech_locs)):
        cur_speech_clip = np.zeros(len(clean))
        cur_noise_clip = np.zeros(len(noise))
        start_idx = max(0, i * chunk_size - fade_overlap)
        end_idx = min(len(clean), (i + 1) * chunk_size + fade_overlap)

        cur_speech_clip[start_idx:end_idx] = clean[start_idx:end_idx].reshape(-1)
        cur_noise_clip[start_idx:end_idx] = noise[start_idx:end_idx].reshape(-1)

        # since may not be exactly divisble on the last one grab all the rest
        if i == (len(speech_locs) - 1):
            cur_speech_clip[start_idx:] = clean[start_idx:].reshape(-1)
            cur_noise_clip[start_idx:] = noise[start_idx:].reshape(-1)

        cur_speech_clip = fade_signal(cur_speech_clip, start_idx, end_idx,
                            chunk_size, fade_overlap, len(speech_locs), i)
        cur_noise_clip = fade_signal(cur_noise_clip, start_idx, end_idx,
                            chunk_size, fade_overlap, len(speech_locs), i)

        room.add_source(speech_locs[i].reshape(-1), signal=cur_speech_clip, delay=0)
        room.add_source(noise_locs[i].reshape(-1), signal=cur_noise_clip, delay=0)

    res = room.simulate(return_premix=True)

    clean_ref = res[np.arange(n_inter) * 2, 0, :].sum(0, keepdims=True)
    noise_ref = res[np.arange(n_inter) * 2 + 1, 0, :].sum(0, keepdims=True)

    if debug:
        import matplotlib.pyplot as plt

        room.plot(freq=[1000, 2000, 4000, 8000], img_order=0)
        plt.savefig('room.png')

        fig, ax = plt.subplots(len(res.sum(0)), 1)
        for i in range(len(ax)):
            ax[i].plot(res.sum(0)[i])
            audiolib.audiowrite(res.sum(0)[i, None].T, cfg['fs'], './debug/res_{}.wav'.format(i))
        plt.savefig('./debug/res.png')

    # soundfile expects an N x C array for multichannel audio
    clean_final = clean_ref.T if cfg['echoic_ref_clean'] else clean.reshape((-1, 1))
    noise_final = noise_ref.T if cfg['echoic_ref_noise'] else noise.reshape((-1, 1))

    return clean_final, noise_final, res.sum(0).T


def simulate_k_room(clean, noise, cfg):
    n_mics = cfg['num_mics']
    room_size = [cfg['room_size']] * 3

    room = pra.ShoeBox(room_size,
                       fs=cfg['fs'],
                       absorption=0.35,
                       max_order=10)

    speech_loc = np.array([cfg['room_size'] - 1] * 3)
    noise_loc = np.array([1] * 3)

    room.add_source(speech_loc, signal=clean, delay=0)
    room.add_source(noise_loc, signal=noise, delay=0)

    mic_locs = np.zeros((3, n_mics))
    mic_locs[:, :cfg['n_noise_mics']] = noise_loc[:, None] + \
        np.random.normal(size=(3, cfg['n_noise_mics']))
    mic_locs[:, cfg['n_noise_mics']:] = speech_loc[:, None] + \
        np.random.normal(size=(3, cfg['n_speech_mics']))

    mic_locs[mic_locs > cfg['room_size']] = cfg['room_size']
    mic_locs[mic_locs < 0] = 0

    mic_array = pra.MicrophoneArray(mic_locs, room.fs)
    room.add_microphone_array(mic_array)

    res = room.simulate(return_premix=True)
    clean_ref = res[0, 0]
    noise_ref = res[1, 0]

    return clean_ref.T, noise_ref.T, res.sum(0).T
