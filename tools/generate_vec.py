# -*- coding: utf-8 -*-
# @Time    : 2020/4/16 22:23
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : generate_vec.py
import argparse
import numpy as np
from tqdm import tqdm
import torch


from framework import factory
from utils import util
from utils.config import Config
from utils.logger import logger, log
from tools.train_embedding import calc_auc, calc_logloss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid'])
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

    model = factory.get_model(cfg)
    model.cuda()
    if cfg.mode == 'valid':
        valid(cfg, model)

def valid(cfg, model):
    criterion = factory.get_loss(cfg)
    util.load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid)
    targets_all = []
    outputs_all = []
    losses = []
    with torch.no_grad():
        model.eval()
        for inputs, targets, ids in tqdm(loader_valid):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())
            losses.append(loss.item())

        result = {
            'targets': np.array(targets_all),
            'outputs': np.array(outputs_all),
            'loss': np.sum(losses) / len(loader_valid),
        }

        result.update(calc_auc(result['targets'], result['outputs']))
        result.update(calc_logloss(result['targets'], result['outputs']))
        result['score'] = result['logloss']

        log(' auc:%.4f micro:%.4f macro:%.4f' % (result['auc'], result['auc_micro'], result['auc_macro']))
        log('%.6f %s' % (result['logloss'], np.round(result['logloss_classes'], 6)))

if __name__ == "__main__":
    main()



