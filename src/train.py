""" 
This is the main entry-point for our code. It trains the mode, makes training plots, saves checkpoints, and stops training.
You call this code with python train.py --cfg <config from train_config.py> --gpu <gpu to run model on>.
"""

import numpy as np
import tqdm
import argparse
import wandb
import os, sys, pathlib
import pprint
import json
from datetime import date, datetime
import pdb
import matplotlib.pyplot as plt
import pickle as pkl
# torch things
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio

# our things
from dataloader import MultiChannelDataset
from enhancer import PonderEnhancer
from loss import get_loss_fn
from plot import generate_plot

def make_wandb_audio(x, caption, fs, is_multi=False):
    if is_multi:
        np_x = x.detach().cpu().numpy().reshape((x.shape[1], -1))
        np_x = np_x / (np.max(np.abs(np_x), axis=1)[:,None] + 1e-7)

        audio_objs = []
        for i in range(min(4, len(np_x))):
            audio_objs.append(wandb.Audio(np_x[i].reshape(-1),
                                            caption=caption + '_chan_{}'.format(i),
                                            sample_rate=fs))

    else:
        np_x = x.detach().cpu().numpy().reshape((-1))
        np_x = np_x / (np.max(np.abs(np_x))+ 1e-7)
        audio_objs = [wandb.Audio(np_x, caption=caption, sample_rate=fs)]

    return audio_objs

def test_sisdr(y_hat, y):
    a = ((y*y_hat).mean(-1) / (y_hat*y_hat).mean(-1)).unsqueeze(-1) * y_hat
    return 10*torch.log10( (a**2).mean(-1) / ((a-y)**2).mean(-1))

def save_dict(data, name, index, start_time, params):
    log_dir = os.path.join(params['log']['log_dir'], start_time)
    os.makedirs(log_dir, exist_ok=True)
    
    filename = os.path.join(log_dir, '{}_{}.pickle'.format(name, index))
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

def run_test_single_dataset(net, test_dataloader, params, data_config, train_index, start_time, 
                            use_wandb=True, wandb_suffix=''):
    net.eval()
    ponder_weight = params['loss']['ponder_weight']
    loss_fn = get_loss_fn(params['loss'])
    fs = data_config['fs']

    total_losses = []
    ponder_losses = []
    enhance_losses = []
    sdrs = []
    per_db_results = {}

    test_index = 0
    audio_objs = []
    for (clean, noise, mix, file_db) in tqdm.tqdm(test_dataloader):
        clean, mix = clean.cuda(), mix.cuda()
        db = file_db[0][:-4]

        # Train
        pred, ponder = net(mix)

        loss_enhance, loss_ponder = loss_fn(clean, pred, ponder)
        total_loss = loss_enhance + ponder_weight * loss_ponder

        if db not in per_db_results:
            per_db_results[db] = {'enhance': [], 'ponder': []}

        per_db_results[db]['enhance'].append(loss_enhance.item())
        per_db_results[db]['ponder'].append(loss_ponder.item())

        total_losses.append(total_loss.item())
        ponder_losses.append(loss_ponder.item())
        enhance_losses.append(loss_enhance.item())

        sdr = test_sisdr(pred, clean)
        sdrs.append(sdr.item())

        if test_index < params['log']['num_audio_save'] and use_wandb:
            audio_objs += make_wandb_audio(clean,
                                           'clean_{}'.format(test_index), fs)
            audio_objs += make_wandb_audio(mix,
                                           'mix_{}'.format(test_index), fs, is_multi=True)
            audio_objs += make_wandb_audio(pred,
                                           'pred_{}'.format(test_index), fs)

        test_index += 1

    if use_wandb:
        wandb.log({'Test Outputs' + wandb_suffix: audio_objs})
        wandb.log({'Total Test Loss' + wandb_suffix: np.array(total_losses).mean()})
        wandb.log({'Ponder Test Loss' + wandb_suffix: np.array(ponder_losses).mean()})
        wandb.log({'Enhance Test Loss' + wandb_suffix: np.array(enhance_losses).mean()})
        wandb.log({'Test SDR' + wandb_suffix: np.array(sdrs).mean()})

        fig_ponder, _, ponder_stats = generate_plot(
            per_db_results, train_index)
        wandb.log({'per_db_ponder' + wandb_suffix: wandb.Image(fig_ponder)})

        fig_enhance, _, enhance_stats = generate_plot(per_db_results, train_index,
                                                      metric='enhance')
        wandb.log({'per_db_enhance' + wandb_suffix: wandb.Image(fig_enhance)})

        save_dict(per_db_results, 'per_db_data' + wandb_suffix, train_index, start_time, params)
        save_dict(ponder_stats, 'per_db_ponder_stats' + wandb_suffix, train_index, start_time, params)
        save_dict(enhance_stats, 'per_db_enhance_stats' + wandb_suffix, train_index, start_time, params)

    net.train()

    return np.array(total_losses).mean(), np.array(ponder_losses).mean()


def run_test(net, params, train_index, start_time, use_wandb=True):
    
    if type(params['test_data_config']) is list:
        all_losses, all_ponders = 0, 0
        for dset_cfg in params['test_data_config']:
            test_dset = get_dataset(dset_cfg)
            test_dataloader = get_dataloader(params, test_dset, train=False)

            wandb_suffix = '_' + pathlib.Path(dset_cfg['output_data_dir']).parents[0].stem
            cur_loss, cur_ponder = run_test_single_dataset(net, 
                                                            test_dataloader, 
                                                            params, 
                                                            dset_cfg,
                                                            train_index, 
                                                            start_time,
                                                            use_wandb=use_wandb,
                                                            wandb_suffix=wandb_suffix)
            all_losses += cur_loss
            all_ponders += cur_ponder

        return all_losses / len(test_dataloader), all_ponders / len(test_dataloader)

    else:
        test_dset = get_dataset(params['test_data_config'])
        test_dataloader = get_dataloader(params, test_dset, train=False)
        return run_test_single_dataset(net, test_dataloader, params, params['test_data_config'], train_index, start_time, use_wandb=use_wandb)

    

def save_model_ckpt(net, params, start_time, index, use_wandb):
    run_dir = os.path.join(params['log']['ckpt_dir'], start_time)
    os.makedirs(run_dir, exist_ok=True)

    filename = os.path.join(run_dir,'{}.pt'.format(index))

    torch.save(net.state_dict(), filename)

    if use_wandb:
        wandb.config.update(net.cpu().state_dict(), allow_val_change=True)
        net.cuda()

def get_dataset(data_config):
    db_lvls = np.linspace(data_config['snr_lower'],
                        data_config['snr_upper'],
                        data_config['total_snrlevels'])

    dset = MultiChannelDataset(root_dir=data_config['output_data_dir'],
                                db_lvls=db_lvls)

    return dset

def get_dataloader(params, dset, train=True):
    batch_size = params['batch_size'] if train else params['test_batch_size']
    num_workers = params['num_workers']

    loader = DataLoader(dset,
                        batch_size=batch_size,
                        shuffle=train,
                        num_workers=num_workers)
    return loader

def setup_wandb(params, net, start_time, str_params_function):
    name = '{}_{}'.format(str_params_function, start_time)
    wandb.init(project="ponder_multi_enhancer", name=name, config=params)
    wandb.watch(net, log=None)

    run_dir = os.path.join(params['log']['ckpt_dir'], start_time)
    os.makedirs(run_dir, exist_ok=True)
    filename_param = os.path.join(run_dir, 'params.json')

    with open(filename_param, 'w') as fp:
        json.dump(params, fp)

    #wandb.save(filename_param)

def check_early_stopping_criterion(losses):
    if len(losses) < 10:
        return False

    # if the overall performance has decreased 2x
    # index 0 holds the test loss and 1 holds the ponder
    if (losses[-2][0] - losses[-1][0]) < 0 and \
        (losses[-3][0] - losses[-2][0]) < 0 and \
        (losses[-4][0] - losses[-3][0]) < 0:
        return True

    return False

def main(params, gpu, start_time, str_params_function, use_wandb=True):
    # get the network ready
    lr = params['opt']['lr']
    grad_clip = params['opt']['grad_clip']
    ponder_weight = 0 if params['loss']['ponder_warmup'] else params['loss']['ponder_weight']

    net = PonderEnhancer(params['model'])

    torch.autograd.set_detect_anomaly(True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    loss_fn = get_loss_fn(params['loss'])

    if use_wandb:
        setup_wandb(params, net, start_time, str_params_function)

    net.cuda()
    net.train()

    stop_early = False
    running_test_losses = []

    index = 0
    print('------- Starting Training -------')
    try:
        for _ in tqdm.tqdm(range(params['epochs'])):
            train_dset = get_dataset(params['train_data_config'])
            train_dataloader = get_dataloader(params, train_dset)

            pbar_cur_loader = tqdm.tqdm(train_dataloader)
            for (clean, _, mix, _) in pbar_cur_loader:
                clean, mix = clean.cuda(), mix.cuda()

                # Train
                opt.zero_grad()
                pred, ponder = net(mix, verbose=False)
                loss_enhance, loss_ponder = loss_fn(clean, pred, ponder)
                total_loss = loss_enhance + ponder_weight * loss_ponder

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt.step()

                index += 1

                if use_wandb:
                    wandb.log({"Total Train Loss": total_loss.item()})
                    wandb.log({"Ponder Train Loss": loss_ponder.item()})
                    wandb.log({"Enhance Train Loss": loss_enhance.item()})

                if index % params['log']['test_pd'] == 0:
                    # regular test that matches training
                    test_loss, test_ponder_loss = run_test(net, params, index, start_time, use_wandb)
                    running_test_losses.append([test_loss, test_ponder_loss])
                    stop_early = check_early_stopping_criterion(running_test_losses)

                if index % params['log']['ckpt_pd'] == 0:
                    save_model_ckpt(net, params, start_time, index, use_wandb)

                if stop_early:
                    save_model_ckpt(net, params, start_time, index, use_wandb)
                    exit()

                if params['loss']['ponder_warmup']:
                    ponder_weight = params['loss']['ponder_weight']
                    ponder_weight *= min(1, index /
                                         (params['loss']['ponder_warmup'] * len(train_dataloader)))

                pbar_cur_loader.set_description("loss : {:10.8f}".format(total_loss.item()))


    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="default")
    parser.add_argument("--gpu", default=0)
    params_function = parser.parse_args().cfg
    gpu = int(parser.parse_args().gpu)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu)

    from train_config import *
    params = globals()[params_function]()

    print('-------Training params-------')
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(params)
    print('---------------------------------')

    start_time = '{}_{}'.format(date.today(),
                                datetime.now().strftime('%H%M%S'))
    main(params, gpu, start_time, params_function, use_wandb=True)
