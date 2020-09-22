import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)