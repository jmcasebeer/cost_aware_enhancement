""" 
This is the dataset object that will load the datasets generates with data_gen.py.
You can test your dataset is working correctly by running this file and listening to the saved outputs.
"""

import glob2
from os.path import join, basename
import numpy as np

import torch
from torch.utils.data import Dataset

import audiolib

class MultiChannelDataset(Dataset):
    def __init__(self, root_dir, db_lvls):
        self.root_dor = root_dir
        self.db_lvls = db_lvls

        self.mono_speech_dir = join(root_dir, 'clean')
        self.mono_noise_dir = join(root_dir, 'noise')
        self.mix_dir = join(root_dir, 'mix')

        mix_files = glob2.glob(join(self.mix_dir, '*.wav'))
        self.file_labels = [(basename(f)[:basename(f).find('_')], basename(f)[basename(f).find('_') + 5:]) for f in mix_files]

    def __len__(self):
        return len(self.file_labels)

    def __getitem__(self, idx):
        (file_idx, file_db_suffix) = self.file_labels[idx]
        file_name = file_idx + '_SNR_' + file_db_suffix

        clean_file = join(self.mono_speech_dir, file_name)
        noise_file = join(self.mono_noise_dir, file_name)
        mix_file = join(self.mix_dir, file_name)

        clean_data, _ = audiolib.audioread(clean_file)
        noise_data, _ = audiolib.audioread(noise_file)
        mix_data, _ = audiolib.audioread(mix_file)

        return torch.Tensor(clean_data), torch.Tensor(noise_data), torch.Tensor(mix_data), file_db_suffix

if __name__=="__main__":
    import data_gen_config
    data_config = data_gen_config.default()

    db_lvls = np.linspace(data_config['snr_lower'],
                        data_config['snr_upper'],
                        data_config['total_snrlevels'])

    dset = MultiChannelDataset(root_dir=data_config['output_data_dir'],
                                db_lvls=db_lvls)

    clean, noise, mix = dset[1]
    audiolib.audiowrite(clean.numpy().T, data_config['fs'], './debug/clean.wav')
    audiolib.audiowrite(noise.numpy().T, data_config['fs'], './debug/noise.wav')
    for i in range(len(mix)):
        audiolib.audiowrite(mix.numpy().T[:,i], data_config['fs'], './debug/mix_{}.wav'.format(i))
