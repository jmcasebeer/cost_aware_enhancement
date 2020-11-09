""" 
This file uses a configuration from data_gen_config.py to generate a dataset. 
It is called directly with python data_gen_config.py --cfg <config name>.
"""

import glob
import numpy as np
import soundfile as sf
import os
import argparse
import tqdm

from audiolib import audioread, audiowrite, snr_setter
from room_sim import simulate_room, simulate_k_room

def make_output_dirs(cfg):
    mixture_dir = os.path.join(cfg['output_data_dir'], 'mix')
    os.makedirs(mixture_dir, exist_ok=True)

    clean_dir = os.path.join(cfg['output_data_dir'], 'clean')
    os.makedirs(clean_dir, exist_ok=True)

    noise_dir = os.path.join(cfg['output_data_dir'], 'noise')
    os.makedirs(noise_dir, exist_ok=True)

    return mixture_dir, clean_dir, noise_dir

def concat_to_size(audio, files, idx, audio_length, silence_length, fs):
    if len(audio) < audio_length * fs:
        while len(audio) <= audio_length * fs:
            idx = idx + 1
            if idx >= np.size(files)-1:
                idx = np.random.randint(0, len(files))

            new_audio, _ = audioread(files[idx])
            audio = np.concatenate((audio, np.zeros(int(fs * silence_length)), new_audio))

    return audio[:audio_length * fs]

def get_speech_and_noise_files(cfg):
    source_speech_dir = cfg["source_speech_dir"]
    assert os.path.exists(source_speech_dir), ("Clean speech data is required - {}".format(source_speech_dir))

    source_noise_dir = cfg["source_noise_dir"]
    assert os.path.exists(source_noise_dir), ("Noise data is required - {}".format(source_noise_dir))

    audio_format = cfg["audio_format"]
    speech_files = glob.glob(os.path.join(source_speech_dir, '*' + audio_format))

    noise_files = []
    noise_files_all = glob.glob(os.path.join(source_noise_dir, '*' + audio_format))
    for noise_file in noise_files_all:
        for noise_type in cfg['noise_types']:
            if os.path.basename(noise_file).startswith(noise_type):
                noise_files.append(noise_file)

    return speech_files, noise_files

def main(cfg):
    mixture_dir, clean_dir, noise_dir = make_output_dirs(cfg)

    audio_format = cfg["audio_format"]
    speech_files, noise_files = get_speech_and_noise_files(cfg)

    snr_lower = cfg["snr_lower"]
    snr_upper = cfg["snr_upper"]
    total_snrlevels = cfg["total_snrlevels"]
    fs = cfg["fs"]
    total_hours = cfg["total_hours"]
    audio_length = cfg["audio_length"]
    silence_length = cfg["silence_length"]

    total_num_mixtures = int(total_hours * 60 * 60 // audio_length)

    for cur_mix_idx in tqdm.tqdm(range(total_num_mixtures)):

        idx_s = np.random.randint(0, np.size(speech_files))
        base_clean, fs = audioread(speech_files[idx_s])
        base_clean = concat_to_size(base_clean, speech_files, idx_s, audio_length, silence_length, fs)

        idx_n = np.random.randint(0, np.size(noise_files))
        base_noise, fs = audioread(noise_files[idx_n])
        base_noise = concat_to_size(base_noise[:len(base_clean)], speech_files, idx_s, audio_length, silence_length, fs)
        base_noise = base_noise[:len(base_clean)]

        for snr_db in np.linspace(snr_lower, snr_upper, total_snrlevels):
            clean, noise = base_clean.copy(), base_noise.copy()
            clean, noise = snr_setter(clean=clean, noise=noise, snr=snr_db)
            
            
            if cfg['room_type'] == 'single_k':
                clean, noise, mixtures = simulate_k_room(clean, noise, cfg)
            elif cfg['room_type'] == 'multi_k':
                cur_cfg = cfg.copy()
                selected_k = np.random.choice(cfg['k_choices'])
                cur_cfg['n_noise_mics'], cur_cfg['n_speech_mics'] = selected_k['n_noise_mics'], selected_k['n_speech_mics']
                clean, noise, mixtures = simulate_k_room(clean, noise, cur_cfg)
            else:
                clean, noise, mixtures = simulate_room(
                    clean, noise, cfg, cur_mix_idx)
            
            
            clean, noise, mixtures = clean[:audio_length * fs], noise[:audio_length * fs], mixtures[:audio_length * fs]

            clean_fname = '{}_SNR_{}'.format(cur_mix_idx, snr_db) + audio_format
            noise_fname = '{}_SNR_{}'.format(cur_mix_idx, snr_db) + audio_format
            mix_fname = '{}_SNR_{}'.format(cur_mix_idx, snr_db) + audio_format

            clean_path = os.path.join(clean_dir, clean_fname)
            noise_path = os.path.join(noise_dir, noise_fname)
            mix_path = os.path.join(mixture_dir, mix_fname)

            audiowrite(clean, fs, clean_path, norm=False)
            audiowrite(noise, fs, noise_path, norm=False)
            audiowrite(mixtures, fs, mix_path, norm=False)


if __name__=="__main__":
    from data_gen_config import *
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default = "default")

    args = vars(parser.parse_args())
    data_config = globals()[args['cfg']]()

    main(data_config)
