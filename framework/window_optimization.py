import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import pretrainedmodels
from mmcv.cnn import constant_init, kaiming_init

from .submodels.contextblock import ContextBlock
from .submodels.asppblock import ASPPModule


WINDOW = OrderedDict({
    'brain': {'center':  40, 'width':   80},
    'subdural': {'center':  80, 'width':  200},
    'tissue': {'center':  40, 'width':  380},
})

def soft_window_to_param(center, width, upbound_value=255., smooth=1.):
    w = 2./width * np.log(upbound_value/smooth - 1.)
    b = -2.*center/width * np.log(upbound_value/smooth - 1.)
    return (w, b)

def window_to_param():
    param = np.array([ soft_window_to_param(w['center'], w['width']) for name, w in  WINDOW.items()])
    weight = param[:,0].tolist()
    bias   = param[:,1].tolist()
    return weight,bias

WEIGHT, BIAS = window_to_param()

class SoftWindow(nn.Module):
    def __init__(self):
        super(SoftWindow, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(3, 1, 1, 1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.zeros(3), requires_grad=True)
        self.weight.data = torch.FloatTensor(WEIGHT).view(3, 1, 1, 1)
        self.bias.data = torch.FloatTensor(BIAS).view(3)

    def forward(self, x):
        x = torch.sigmoid(F.conv2d(x, self.weight, self.bias))
        return x

class Without_Window_Optimization(nn.Module):
    def __init__(self, model, n_output):
        super(Without_Window_Optimization, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(
            model.last_linear.in_features,
            n_output,
        )

    def forward(self, x):
        x = self.resnet_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        logit = self.last_linear(x)

        return logit, x

class Semi_Auto_Window_Optimization(nn.Module):
    def __init__(self, model, n_output):
        super(Semi_Auto_Window_Optimization, self).__init__()

        self.window = SoftWindow()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.bn = nn.BatchNorm2d(3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(
            model.last_linear.in_features,
            n_output,
        )

    def forward(self, x):
        x = self.window(x)
        x = self.bn(x)
        x = self.resnet_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        logit = self.last_linear(x)

        return logit, x

#v1
# class Auto_Window_Optimization(nn.Module):
#     def __init__(self, model, n_output):
#         super(Auto_Window_Optimization, self).__init__()
#
#         self.pre_features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(1, 3, 7, stride=2)),
#             ('bn1', nn.BatchNorm2d(3)),
#             ('relu1', nn.ReLU()),
#             ('pool1', nn.AvgPool2d(2)),
#             ('conv2', nn.Conv2d(3, 16, 5, stride=2)),
#             ('bn2', nn.BatchNorm2d(16)),
#             ('relu2', nn.ReLU()),
#             ('pool2', nn.AvgPool2d(2)),
#             ('conv3', nn.Conv2d(16, 512, 3, stride=2)),
#             ('bn3', nn.BatchNorm2d(512)),
#             ('relu3', nn.ReLU()),
#             ('pool3', nn.AdaptiveAvgPool2d(1)),
#         ]))
#
#         # self.pre_features = nn.Sequential(OrderedDict([
#         #     ('conv1', nn.Conv2d(1, 3, 7, stride=2)),
#         #     ('bn1', nn.BatchNorm2d(3)),
#         #     ('relu1', nn.ReLU()),
#         #     ('pool1', nn.AvgPool2d(2)),
#         #     ('conv2', nn.Conv2d(3, 16, 5, stride=2)),
#         #     ('bn2', nn.BatchNorm2d(16))
#         # ]))
#         #
#         # self.post_features = nn.Sequential(OrderedDict([
#         #     ('pool2', nn.AvgPool2d(2)),
#         #     ('conv3', nn.Conv2d(512, 1024, 3, stride=2)),
#         #     ('bn3', nn.BatchNorm2d(1024)),
#         #     ('relu3', nn.ReLU()),
#         #     ('pool3', nn.AdaptiveAvgPool2d(1)),
#         # ]))
#
#         # self.transition = nn.Sequential(OrderedDict([
#         #     ('conv1', nn.Conv2d(80, 512, 3, stride=2)),
#         #     ('bn1', nn.BatchNorm2d(512)),
#         #     ('relu1', nn.ReLU()),
#         #     ('pool1', nn.AvgPool2d(2)),
#         #     ('conv2', nn.Conv2d(512, 512, 3, stride=2)),
#         #     ('bn2', nn.BatchNorm2d(512)),
#         #     ('relu2', nn.ReLU()),
#         #     ('pool2', nn.AdaptiveAvgPool2d(1)),
#         # ]))
#
#         self.WL_predict = nn.Linear(512, 3)
#         self.WW_predict = nn.Linear(512, 3)
#
#         # self.context_block = ContextBlock(inplanes=16, ratio=1/4)
#         self.aspp_block = ASPPModule(features=16, inner_features=256, out_features=512, dilations=(12, 24, 36))
#
#
#         self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
#         self.bn = nn.BatchNorm2d(3)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.last_linear = nn.Linear(
#             model.last_linear.in_features,
#             n_output,
#         )
#
#         #初始化
#         for layer in self.pre_features:
#             if isinstance(layer, nn.Conv2d):
#                 kaiming_init(layer)
#             elif isinstance(layer, nn.BatchNorm2d):
#                 constant_init(layer, 1)
#
#         # for layer in self.post_features:
#         #     if isinstance(layer, nn.Conv2d):
#         #         kaiming_init(layer)
#         #     elif isinstance(layer, nn.BatchNorm2d):
#         #         constant_init(layer, 1)
#
#         # for layer in self.transition:
#         #     if isinstance(layer, nn.Conv2d):
#         #         kaiming_init(layer)
#         #     elif isinstance(layer, nn.BatchNorm2d):
#         #         constant_init(layer, 1)
#
#         nn.init.normal_(self.WL_predict.weight, 0, 0.005)
#         nn.init.constant_(self.WL_predict.bias, -0)
#         nn.init.normal_(self.WW_predict.weight, 0, 0.005)
#         nn.init.constant_(self.WW_predict.bias, -2)
#
#     def forward(self, x):
#         origin = x
#         x = self.pre_features(x)
#
#         #two branch
#         # context_features = self.context_block(x)
#         # aspp_features = self.aspp_block(x)
#         #
#         # context_features = context_features + x
#         # features = torch.cat([context_features, aspp_features], dim=1)
#         #
#         # features = self.transition(features)
#         # features = features.view(1, features.shape[0], -1)
#         # features = torch.mean(features, dim=1)
#
#
#         #learning window parameters
#         # WL = self.WL_predict(features)
#         # WW = self.WW_predict(torch.cat([features, WL], dim=1))
#
#         x = x.view(1, x.shape[0], -1)
#         x = torch.mean(x, dim=1)
#         WL = self.WL_predict(x)
#         WW = self.WW_predict(x)
#
#         self.weight = torch.nn.Parameter(WL.view(3, 1, 1, 1), requires_grad=True)
#         self.bias = torch.nn.Parameter(WW.view(3), requires_grad=True)
#
#
#         #have learned window parameters
#         x = torch.sigmoid(F.conv2d(origin, self.weight, self.bias))
#         x = self.bn(x)
#         x = self.resnet_layer(x)
#         x = self.avg_pool(x)
#         x = x.view(x.shape[0], -1)
#         logit = self.last_linear(x)
#
#         return logit

def Statistic(is_training, X, gamma, beta, moving_mean, momentum):
    if not is_training:
        X_hat = moving_mean
    else:
        mean = X.mean(dim=0)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        X_hat = mean
    Y = gamma * X_hat + beta
    return Y, moving_mean


class Auto_Window_Optimization(nn.Module):
    def __init__(self, model, n_output):
        super(Auto_Window_Optimization, self).__init__()

        self.pre_features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 3, 7, stride=2)),
            ('bn1', nn.BatchNorm2d(3)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.AvgPool2d(2)),
            ('conv2', nn.Conv2d(3, 16, 5, stride=2)),
            ('bn2', nn.BatchNorm2d(16)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.AvgPool2d(2)),
            ('conv3', nn.Conv2d(16, 512, 3, stride=2)),
            ('bn3', nn.BatchNorm2d(512)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.AdaptiveAvgPool2d(1)),
        ]))

        self.WL_predict = nn.Linear(512, 3)
        self.WW_predict = nn.Linear(512, 3)

        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])
        self.bn = nn.BatchNorm2d(3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(
            model.last_linear.in_features,
            n_output,
        )

        #初始化
        for layer in self.pre_features:
            if isinstance(layer, nn.Conv2d):
                kaiming_init(layer)
            elif isinstance(layer, nn.BatchNorm2d):
                constant_init(layer, 1)

        nn.init.normal_(self.WL_predict.weight, 0, 0.005)
        nn.init.constant_(self.WL_predict.bias, -0)
        nn.init.normal_(self.WW_predict.weight, 0, 0.005)
        nn.init.constant_(self.WW_predict.bias, -2)

        self.WW_gamma = nn.Parameter(torch.ones((1, 3)))
        self.WW_beta = nn.Parameter(torch.zeros((1, 3)))
        self.WL_gamma = nn.Parameter(torch.ones((1, 3)))
        self.WL_beta = nn.Parameter(torch.zeros((1, 3)))

        self.WW_moving = torch.zeros((1, 3)).cuda()
        self.WL_moving = torch.zeros((1, 3)).cuda()

    def forward(self, x):
        # if self.WW_moving.device != x.device:
        #     self.WW_moving = self.WW_moving.to(x.device)
        #     self.WL_moving = self.WL_moving.to(x.device)

        origin = x
        x = self.pre_features(x)

        #直接得出
        # x = x.view(1, x.shape[0], -1)
        # x = torch.mean(x, dim=1)
        # WL = self.WL_predict(x)
        # WW = self.WW_predict(x)

        #滑动平均
        x = x.view(x.shape[0], -1)
        WL = self.WL_predict(x)
        WW = self.WW_predict(x)
        WL, self.WL_moving = Statistic(self.training, WL, self.WL_gamma, self.WL_beta, self.WL_moving, 0.9)
        WW, self.WW_moving = Statistic(self.training, WW, self.WW_gamma, self.WW_beta, self.WW_moving, 0.9)
        weight = WL.view(3, 1, 1, 1)
        bias = WW.view(3)


        #have learned window parameters
        x = torch.sigmoid(F.conv2d(origin, weight, bias))
        x = self.bn(x)
        x = self.resnet_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        logit = self.last_linear(x)

        return logit, x


if __name__ == "__main__":

    model_func = pretrainedmodels.__dict__["se_resnext50_32x4d"]
    model = model_func(num_classes=1000, pretrained="imagenet")
    model = Net(model, n_output=6)
    input = np.random.uniform(-1, 1, (1, 1, 512, 512))
    input = torch.from_numpy(input).float()
    output = model(input)
    print(output.shape)
    # print(list(model.window.parameters()))