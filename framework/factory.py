import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensor
import pretrainedmodels


from .dataset.custom_dataset import CustomDataset
from .dataset.seq_dataset import SeqDataset, collate_fn
from .transforms.transforms import RandomResizedCrop, RandomDicomNoise
from utils.logger import log

from .sequence.seq import Sequence
from .window_optimization import Semi_Auto_Window_Optimization, Auto_Window_Optimization, Without_Window_Optimization

def get_loss(cfg):
    #loss = getattr(nn, cfg.loss.name)(**cfg.loss.params)
    loss = getattr(nn, cfg.loss.name)(weight=torch.FloatTensor([2,1,1,1,1,1]).cuda(), **cfg.loss.params)
    log('loss: %s' % cfg.loss.name)
    return loss

def get_dataloader(cfg, folds=None):
    dataset = CustomDataset(cfg, folds)
    log('use default(random) sampler')
    loader = DataLoader(dataset, **cfg.loader)
    return loader

def get_seq_dataloader(cfg, folds=None):
    dataset = SeqDataset(embedding_dir=cfg.embedding_dir,annotations=cfg.annotations, folds=folds)
    log('use custom sampler')
    loader = DataLoader(dataset, **cfg.loader, collate_fn=collate_fn)
    return loader


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(A, transform.name):
            return getattr(A, transform.name)
        else:
            return eval(transform.name)
    transforms = [get_object(transform)(**transform.params) for transform in cfg.transforms]
    return A.Compose(transforms)


def get_model(cfg):
    log(f'auto:{cfg.auto}')
    log(f'soft_window: {cfg.soft_window}')
    log(f'model: {cfg.model.name}')
    log(f'pretrained: {cfg.model.pretrained}')
    try:
        model_func = pretrainedmodels.__dict__[cfg.model.name]
    except KeyError as e:
        model_func = eval(cfg.model.name)

    model = model_func(num_classes=1000, pretrained=cfg.model.pretrained)

    if cfg.soft_window == False:
        model = Without_Window_Optimization(model, n_output=cfg.model.n_output)
    elif cfg.soft_window == True and cfg.auto == False:
        model = Semi_Auto_Window_Optimization(model, n_output=cfg.model.n_output)
    elif cfg.soft_window == True and cfg.auto == True:
        model = Auto_Window_Optimization(model, n_output=cfg.model.n_output)

    return model

def get_seq_model(cfg):
    log(f'model: {cfg.model.name}')
    model = Sequence(cfg.model.input_size,
                     cfg.model.hidden_size,
                     cfg.model.num_layers,
                     cfg.model.n_output)
    return model


def get_optim(cfg, parameters):
    optim = getattr(torch.optim, cfg.optim.name)(parameters, **cfg.optim.params)
    log(f'optim: {cfg.optim.name}')
    return optim


def get_scheduler(cfg, optim, last_epoch):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optim,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optim,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log(f'last_epoch: {last_epoch}')
    return scheduler