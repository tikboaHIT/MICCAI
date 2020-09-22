# -*- coding: utf-8 -*-
# @Time    : 2020/4/26 20:48
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : train_sequence.py
import sys
import os
import time
import argparse
import random
import collections
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, log_loss

import torch
from torch import nn
import torch.nn.functional as F

from framework import factory
from utils import util
from utils.config import Config
from utils.logger import logger, log
from tqdm import tqdm
from tools.train_embedding import calc_auc, calc_logloss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'generate'])
    parser.add_argument('config')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()

def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.fold = args.fold
    cfg.gpu = args.gpu

    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    torch.cuda.set_device(cfg.gpu)
    util.set_seed(cfg.seed)

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')

    model = factory.get_seq_model(cfg)
    model.cuda()
    if cfg.mode == 'train':
        train(cfg, model)
    if cfg.mode == 'valid':
        valid(cfg, model)

def valid(cfg, model):
    criterion = factory.get_loss(cfg)
    util.load_model(cfg.snapshot, model)
    loader_valid = factory.get_seq_dataloader(cfg.data.valid, [cfg.fold] if cfg.fold is not None else None)
    with torch.no_grad():
        results = run_nn(cfg, 'valid', model, loader_valid, criterion=criterion)

def train(cfg, model):
    criterion = factory.get_loss(cfg)
    optim = factory.get_optim(cfg, model.parameters())
    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }
    if cfg.resume_from:
        detail = util.load_model(cfg.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })

    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    loader_train = factory.get_seq_dataloader(cfg.data.train, folds)
    loader_valid = factory.get_seq_dataloader(cfg.data.valid, [cfg.fold])

    log('train data: loaded %d records' % len(loader_train.dataset))
    log('valid data: loaded %d records' % len(loader_valid.dataset))

    scheduler = factory.get_scheduler(cfg, optim, best['epoch'])

    for epoch in range(best['epoch'] + 1, cfg.epoch):

        log(f'\n----- epoch {epoch} -----')
        run_nn(cfg, 'train', model, loader_train, criterion=criterion, optim=optim)
        with torch.no_grad():
            val = run_nn(cfg, 'valid', model, loader_valid, criterion=criterion)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)

        util.save_model(model, optim, detail, cfg.fold, cfg.workdir)

        log('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))
        scheduler.step()

def run_nn(cfg, mode, model, loader, criterion=None, optim=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise RuntimeError('Unexpected mode %s' % mode)

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for i, (data, target_data, targets, lengths, seq_positions_list, target_position_list) in enumerate(loader):
        data = data.cuda()
        target_data = target_data.cuda()
        targets = targets.cuda()
        outputs = model(data, target_data, lengths, seq_positions_list, target_position_list)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if None:
                pass
            else:
                loss.backward()  # accumulate loss
            if (i + 1) % cfg.data.train.n_grad_acc == 0:
                optim.step()  # update
                optim.zero_grad()  # flush

        with torch.no_grad():
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())
            # outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i + 1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses) / (i + 1)):.6f} loss200:{(np.sum(losses[-200:]) / (min(i + 1, 200))):.6f} lr:{util.get_lr(optim):.2e}'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i + 1),
    }

    if mode in ['train', 'valid']:
        result.update(calc_auc(result['targets'], result['outputs'], cfg))
        result.update(calc_logloss(result['targets'], result['outputs']))
        result['score'] = result['logloss']

        log(progress + ' auc:%.4f micro:%.4f macro:%.4f' % (result['auc'], result['auc_micro'], result['auc_macro']))
        log('%.6f %s' % (result['logloss'], np.round(result['logloss_classes'], 6)))
    else:
        log('')

    return result

if __name__ == "__main__":
    main()