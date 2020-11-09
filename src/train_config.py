""" 
This is where you specify model params and which dataset to use. You will call these configs from train.py.
"""
import data_gen_config

def complex_ponder_med_5e5_mostdb_5_warmup():
    hyp = {
        'train_data_config': data_gen_config.med_train_most_db(),
        'test_data_config': data_gen_config.med_test_most_db(),

        'opt': {
            'lr': 1e-3,
            'grad_clip': 1,
        },

        'log': {
            'ckpt_pd': 5000,
            'test_pd': 5000,
            'num_audio_save': 3,
            'ckpt_dir': '../ckpts',
            'log_dir': '../logs'
        },

        'model': {
            'stft_win_size': 512,
            'stft_hop': 128,
            'hidden_size': 256,
            'n_heads': 4,
            'dropout': 0.25,
            'complex_mask': True,
            'two_d_convs': False,

            'encoder_params': {
                'n_layers': 2,
                'in_channels': [512 // 2 + 1] * 2,
                'out_channels': [512 // 2 + 1] * 2,
                'kernel_size': [5] * 2,
                'stride': [1] * 2,
                'padding': [2] * 2,
                'dilation': [1] * 2,
            },

            'decoder_params': {
                'n_layers': 4,
                'in_channels': [512 // 2 + 1] * 4,
                'out_channels': [512 // 2 + 1] * 4,
                'kernel_size': [5] * 4,
                'stride': [1] * 4,
                'padding': [2] * 4,
                'dilation': [1] * 4,
            }
        },

        'loss': {
            'enhance_loss_fn': 'power_mag_stft_l1',
            'ponder_weight': 5e-5,
            'ponder_warmup': 5,
        },


        'batch_size': 16,
        'test_batch_size': 1,
        'num_workers': 10,
        'epochs': 100,
    }

    return hyp

def complex_ponder_med_k_9_mostdb_5_warmup():
    hyp = {
        'train_data_config': data_gen_config.med_train_most_db(),
        'test_data_config': data_gen_config.med_test_most_db(),

        'opt': {
            'lr': 1e-3,
            'grad_clip': 1,
        },

        'log': {
            'ckpt_pd': 5000,
            'test_pd': 5000,
            'num_audio_save': 3,
            'ckpt_dir': '../ckpts',
            'log_dir': '../logs'
        },

        'model': {
            'stft_win_size': 512,
            'stft_hop': 128,
            'hidden_size': 256,
            'n_heads': 4,
            'dropout': 0.25,
            'complex_mask': True,
            'two_d_convs': False,
            'fixed_k': 9,

            'encoder_params': {
                'n_layers': 2,
                'in_channels': [512 // 2 + 1] * 2,
                'out_channels': [512 // 2 + 1] * 2,
                'kernel_size': [5] * 2,
                'stride': [1] * 2,
                'padding': [2] * 2,
                'dilation': [1] * 2,
            },

            'decoder_params': {
                'n_layers': 4,
                'in_channels': [512 // 2 + 1] * 4,
                'out_channels': [512 // 2 + 1] * 4,
                'kernel_size': [5] * 4,
                'stride': [1] * 4,
                'padding': [2] * 4,
                'dilation': [1] * 4,
            }
        },

        'loss': {
            'enhance_loss_fn': 'power_mag_stft_l1',
            'ponder_weight': 0,
            'ponder_warmup': 0,
        },


        'batch_size': 16,
        'test_batch_size': 1,
        'num_workers': 10,
        'epochs': 100,
    }

    return hyp
