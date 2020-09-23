# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 17:24
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : train_embedding.py
import sys
import os
import time
import argparse
import random
import collections
import pickle

import pandas as pd
import numpy as np
import sklearn.metrics as m

import torch
from torch import nn
import torch.nn.functional as F

from framework import factory
from utils import util
from utils.config import Config
from utils.logger import logger, log
from utils import mappings
from tqdm import tqdm
# import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'generate'])
    parser.add_argument('config')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output')
    return parser.parse_args()

def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.fold = args.fold
    cfg.output = args.output
    cfg.gpu = args.gpu

    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    torch.cuda.set_device(cfg.gpu)
    util.set_seed(cfg.seed)

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')

    model = factory.get_model(cfg)
    model.cuda()
    if cfg.mode == 'train':
        train(cfg, model)
    if cfg.mode == 'valid':
        valid(cfg, model)
    if cfg.mode == "generate":
        generate(cfg, model)


def generate(cfg, model):
    criterion = factory.get_loss(cfg)
    util.load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid, [fold for fold in range(cfg.n_fold)])

    targets_all = []
    outputs_all = []
    ids_all = []
    losses = []
    model.eval()
    with torch.no_grad():
        for inputs, targets, ids in tqdm(loader_valid):
            inputs = inputs.cuda()
            outputs, embeddings = model(inputs)

            embeddings = embeddings.cpu().numpy()

            #save embedding
            for index, id in enumerate(ids):
                embedding = embeddings[index, :]
                path = "/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/embeddings/{}".format(id)
                np.save(path, embedding)

def valid(cfg, model):
    criterion = factory.get_loss(cfg)
    util.load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold] if cfg.fold is not None else None)
    with torch.no_grad():
        results = run_nn(cfg, 'valid', model, loader_valid, criterion=criterion)


def train(cfg, model):
    criterion = factory.get_loss(cfg)
    optim = factory.get_optim(cfg, model.parameters())

    # parameters_group = [{'params': model.pre_features.parameters(), 'lr': cfg.optim["params"]["lr"] * 5},
    #                     {'params': model.WL_predict.parameters(), 'lr': cfg.optim["params"]["lr"]},
    #                     {'params': model.WW_predict.parameters(), 'lr': cfg.optim["params"]["lr"]},
    #
    #                     {'params': model.resnet_layer.parameters(), 'lr': cfg.optim["params"]["lr"]},
    #                     {'params': model.bn.parameters(), 'lr': cfg.optim["params"]["lr"]},
    #                     {'params': model.last_linear.parameters(), 'lr': cfg.optim["params"]["lr"]}]
    #
    # optim = factory.get_optim(cfg, parameters_group)

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
    loader_train = factory.get_dataloader(cfg.data.train, folds)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])

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

    for i, (inputs, targets, ids) in enumerate(loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs, embeddings = model(inputs)

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
            ids_all.extend(ids)
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())
            # outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
        progress = f'\r[{mode}] {i + 1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses) / (i + 1)):.6f} loss200:{(np.sum(losses[-200:]) / (min(i + 1, 200))):.6f} lr:{util.get_lr(optim):.2e}'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': ids_all,
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


def calc_logloss(targets, outputs, eps=1e-5):
    # for RSNA
    try:
        logloss_classes = [m.log_loss(np.round(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
    except ValueError as e:
        logloss_classes = [1, 1, 1, 1, 1, 1]

    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
    }

# def calc_auc(targets, outputs):
#     macro = m.roc_auc_score(targets, np.round(outputs), average='macro')
#     micro = m.roc_auc_score(targets, np.round(outputs), average='micro')
#     print(m.classification_report(targets, np.round(outputs)))
#     return {
#         'auc': (macro + micro) / 2,
#         'auc_macro': macro,
#         'auc_micro': micro,
#     }

def calc_auc(targets, outputs, cfg):
    macro = m.roc_auc_score(targets, outputs, average='macro')
    micro = m.roc_auc_score(targets, outputs, average='micro')
    print(m.roc_auc_score(targets, outputs, average=None))
    plot_roc(targets, outputs, cfg)
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }

def plot_roc(targets, outputs, cfg):
    n_classes = targets.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = m.roc_curve(targets[:, i], outputs[:, i])
        roc_auc[i] = m.auc(fpr[i], tpr[i])

    # plt.figure(figsize=(12, 10))
    # for i in range(n_classes):
    #     plt.subplot(2, 3, i+1)
    #     lw = 2
    #     plt.plot(fpr[i], tpr[i], color='darkorange',
    #              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('{}'.format(mappings.num_to_label[i]))
    #     plt.legend(loc="lower right")
    # plt.savefig(cfg.workdir+"/auc_all_classes.png")

if __name__ == "__main__":
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    main()