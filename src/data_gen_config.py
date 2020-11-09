""" 
This file contains configurations for generating train and test datasets. 
The configurations defined here are specified as command line arguments in data_gen.py.
"""

def med_train_most_db():
    hyp = {
        'fs': 16000,
        'audio_format': '.wav',
        'audio_length': 2,
        'silence_length': 0.25,
        'total_hours': 5,
        'snr_lower': -10,
        'snr_upper': 10,
        'total_snrlevels': 9,

        'num_mics': 15,
        'room_type': 'rand',
        'n_inter_locs': 5,
        'room_size_upper': 15,
        'room_size_lower': 10,
        'room_geom_fixed': False,
        'echoic_ref_clean': True,
        'echoic_ref_noise': True,

        'source_noise_dir': '/mnt/data/ms-snsd/MS-SNSD-master/noise_train',
        'source_speech_dir': '/mnt/data/ms-snsd/MS-SNSD-master/clean_train',
        'output_data_dir': '/mnt/data/ms-snsd/multi_channel/med_15_mics_most_db/train',

        'noise_types': ['AirConditioner', 'Babble', 'Bus', 'Car', 'CopyMachine',
                        'Munching', 'ShuttingDoor', 'SqueakyChair', 'Traffic',
                        'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing'],
    }

    return hyp

def med_test_most_db():
    hyp = {
        'fs': 16000,
        'audio_format': '.wav',
        'audio_length': 2,
        'silence_length': 0.25,
        'total_hours': 1,
        'snr_lower': -10,
        'snr_upper': 10,
        'total_snrlevels': 9,

        'num_mics': 15,
        'room_type': 'rand',
        'n_inter_locs': 5,
        'room_size_upper': 15,
        'room_size_lower': 10,
        'room_geom_fixed': False,
        'echoic_ref_clean': True,
        'echoic_ref_noise': True,

        'source_noise_dir': '/mnt/data/ms-snsd/MS-SNSD-master/noise_test',
        'source_speech_dir': '/mnt/data/ms-snsd/MS-SNSD-master/clean_test',
        'output_data_dir': '/mnt/data/ms-snsd/multi_channel/med_15_mics_most_db/test',

        'noise_types': ['AirConditioner', 'Babble', 'Bus', 'Car', 'CopyMachine',
                        'Munching', 'ShuttingDoor', 'SqueakyChair', 'Traffic',
                        'Typing', 'VacuumCleaner', 'WasherDryer', 'Washing'],
    }

    return hyp