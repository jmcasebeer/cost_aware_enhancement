""" 
This file is called after a model has been trained. It takes a model checkpoint from the ckpt directory and runs it on 
a specified data config. We used this file to make our STOI plots and it can be called with:
python run_stoi.py --cfg <config from train_config> --date <start time of the model ckpt> 
--epoch <epoch of the model ckpt> --gpu <gpu to this test on>.

All results are then saved to that models logs directory.
"""

import os
import argparse
import librosa
import pystoi
import tqdm
import pprint
import numpy as np

import torch

from enhancer import PonderEnhancer
from train import get_dataloader, get_dataset, get_loss_fn, save_dict


def get_model_ckpt(params, date, epoch):
    run_dir = os.path.join(params['log']['ckpt_dir'], date)
    filename = os.path.join(run_dir,'{}.pt'.format(epoch))
    return filename

def main(params, date, epoch, gpu):
    net = PonderEnhancer(params['model'])
    ckpt = get_model_ckpt(params, date, epoch)
    net.load_state_dict(torch.load(ckpt))

    net.cuda()
    net.eval()

    # run  test 
    test_dset = get_dataset(params['test_data_config'])
    test_dataloader = get_dataloader(params, test_dset, train=False)
    loss_fn = get_loss_fn(params['loss'])
    fs = params['test_data_config']['fs']

    per_db_results = {}
    for (clean, noise, mix, file_db) in tqdm.tqdm(test_dataloader):
        clean, mix = clean.cuda(), mix.cuda()
        db = file_db[0][:-4]

        # Train
        pred, ponder = net(mix, verbose=False)  # change debug -> verbose
        _, loss_ponder = loss_fn(clean, pred, ponder)

        if db not in per_db_results:
            per_db_results[db] = {'enhance': [], 'ponder': []}

        # get the perceptual metrics
        np_clean = clean.detach().cpu().numpy().reshape(-1)
        np_pred = pred.detach().cpu().numpy().reshape(-1)
        
        per_db_results[db]['enhance'].append(pystoi.stoi(clean, enhanced, sr))
        per_db_results[db]['ponder'].append(loss_ponder.item())

    # save it all
    save_dict(per_db_results, 'stoi', epoch, date, params)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="default")
    parser.add_argument("--date", default="")
    parser.add_argument("--epoch", default=0)
    parser.add_argument("--gpu", default=0)

    params_function = parser.parse_args().cfg
    gpu = int(parser.parse_args().gpu)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    from train_config import *
    params = globals()[params_function]()

    print('-------Training params-------')
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(params)
    print('---------------------------------')

    main(params, parser.parse_args().date, parser.parse_args().epoch, gpu)
