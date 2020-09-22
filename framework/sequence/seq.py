# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 16:35
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : seq.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from collections import OrderedDict
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from torch.utils.data import DataLoader
from framework.dataset.seq_dataset import SeqDataset, collate_fn
import torch.nn.functional as F

class Sequence(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_output):
        super(Sequence, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc_target = nn.Sequential(OrderedDict([
            ('line1', nn.Linear(input_size, hidden_size)),
            #TODO: fc_target加层
            ('relu1', nn.ReLU()),
            ('line2', nn.Linear(hidden_size, hidden_size)),
            ('relu2', nn.ReLU()),
        ]))

        self.last_linear = nn.Sequential(OrderedDict([
            ('line1', nn.Linear(hidden_size*2, hidden_size)),
            ('relu1', nn.ReLU()),
            # TODO: dropout
            #('drop1', nn.Dropout(0.5)),
            ('line2', nn.Linear(hidden_size, n_output)),
        ]))


    def forward(self, data, target_data, lengths, seq_positions_list, target_position_list):
        #slice data
        data = rnn_utils.pack_padded_sequence(data, lengths, batch_first=True)
        out, _ = self.lstm(data)
        output, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        slice_seq = output.data

        #slice target
        slice_target = self.fc_target(target_data)

        #calculate dependencies
        features = []
        for index, length in enumerate(lengths):
            curr_slice_seq = slice_seq[index, :length, :] #抽取对应位置和长度的切片序列
            # TODO: 切片序列归一化
            curr_slice_seq = F.normalize(curr_slice_seq, p=2, dim=1)
            curr_slice_target = slice_target[index, :].unsqueeze(1)

            #soft attention: context similarity
            weight = curr_slice_seq.mm(curr_slice_target)
            # TODO: 权重归一化
            #weight = F.softmax(weight, dim=0)
            weight_curr_slice_seq = weight.mul(curr_slice_seq)
            soft_mean_curr_slice_seq = torch.sum(weight_curr_slice_seq, dim=0)/torch.sum(weight)

            #hard attention: spatial similarity
            seq_positions = torch.from_numpy(np.array(seq_positions_list[index]))
            target_position = torch.from_numpy(np.array(target_position_list[index]))
            weight = torch.exp(-1*((seq_positions-target_position)**2)/(seq_positions.size(0)**0.2))
            weight = F.softmax(weight, dim=0).unsqueeze(1).float().cuda()
            weight_curr_slice_seq = weight.mul(curr_slice_seq)
            hard_mean_curr_slice_seq = torch.sum(weight_curr_slice_seq, dim=0) / torch.sum(weight)

            curr_slice_target = curr_slice_target.squeeze(1)
            mean_curr_slice_seq = soft_mean_curr_slice_seq + hard_mean_curr_slice_seq
            feature = torch.cat((mean_curr_slice_seq, curr_slice_target))
            features.append(feature)

        #generate features
        new_features = features[0].unsqueeze(0)
        for feature in features[1:]:
            new_features = torch.cat((new_features, feature.unsqueeze(0)))

        #classification
        out = self.last_linear(new_features)

        return out


if __name__ == "__main__":
    dataset = SeqDataset(embedding_dir="/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/embeddings",
                      annotations="/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/train_folds8_seed300.pkl",
                      folds=[0])

    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=8,
                            collate_fn=collate_fn)

    input_size = 2048
    hidden_size = 1024
    num_layers = 1
    batch_size = 2
    n_output = 6

    model = Sequence(input_size, hidden_size, num_layers, n_output)
    criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([2, 1, 1, 1, 1, 1]))
    optimizer = torch.optim.Adam(lr=1.0e-4, params=model.parameters())

    iteration = 0
    for data, target_data, targets, lengths in dataloader:
        outputs = model(data, target_data, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())

