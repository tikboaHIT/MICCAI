# -*- coding: utf-8 -*-
# @Time    : 2020/4/24 11:08
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : seq_dataset.py
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
import numpy as np
import os.path as osp
from utils import mappings, misc
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm

class SeqDataset(Dataset):
    def __init__(self, embedding_dir, annotations, folds):
        with open(annotations, 'rb') as f:
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]
            print('read dataset (%d records)' % len(self.df))

        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ID = row["ID"]
        PatientID = row["SeriesInstanceUID"]
        seq_frame = self.df[self.df["SeriesInstanceUID"]==PatientID].sort_values("PositionOrd")
        seq_frame = seq_frame[seq_frame["ID"] != ID]

        #CT scans邻近切片向量
        seq_ids = seq_frame["ID"].tolist()
        start_id = seq_ids[0]
        slice_path = osp.join(self.embedding_dir, "{}.npy".format(start_id))
        embedding = np.load(slice_path)[None]
        for id in seq_ids[1::]:
            slice_path = osp.join(self.embedding_dir, "{}.npy".format(id))
            slice_embedding = np.load(slice_path)[None]
            embedding = np.vstack((embedding, slice_embedding))
        embedding = torch.from_numpy(embedding)

        #目标切片向量
        target_slice_path = osp.join(self.embedding_dir, "{}.npy".format(ID))
        target_embedding = np.load(target_slice_path)[None]
        target_embedding = torch.from_numpy(target_embedding)

        target = np.array([0.0] * len(mappings.label_to_num))
        for label in row.labels.split():
            cls = mappings.label_to_num[label]
            target[cls] = 1.0
        target = torch.FloatTensor(target)

        #提取CT scans位置信息
        seq_positions = seq_frame["PositionOrd"].tolist()
        target_position = row["PositionOrd"]

        return {
            'embedding': embedding,
            'target_embedding': target_embedding,
            'target': target,
            'seq_positions':seq_positions,
            'target_positions': target_position
        }

def collate_fn(train_data):
    embedding_datas = []
    target_embeddings = []
    targets = []
    seq_positions_list = []
    target_position_list = []

    for data in train_data:
        embedding, target_embedding, target, seq_positions, target_position = data.values()
        embedding_datas.append(embedding)
        target_embeddings.append((target_embedding, embedding.size(0)))
        targets.append((target, embedding.size(0)))
        seq_positions_list.append((seq_positions, embedding.size(0)))
        target_position_list.append((target_position, embedding.size(0)))

    #slice embedding pad
    embedding_datas.sort(key=lambda data: data.size(0), reverse=True)
    data_length = [data.size(0) for data in embedding_datas]
    embedding_datas = rnn_utils.pad_sequence(embedding_datas, batch_first=True, padding_value=0)

    #adjust target embeddings
    target_embeddings.sort(key=lambda data: data[1], reverse=True)
    new_target_embeddings = target_embeddings[0][0]
    for embedding, _ in target_embeddings[1:]:
        new_target_embeddings = torch.cat((new_target_embeddings, embedding), dim=0)

    #adjust target
    targets.sort(key=lambda data: data[1], reverse=True)
    new_targets = targets[0][0].unsqueeze(0)
    for target, _ in targets[1:]:
        new_targets = torch.cat((new_targets, target.unsqueeze(0)), dim=0)

    #adjust seq_positions
    seq_positions_list.sort(key=lambda data: data[1], reverse=True)
    new_seq_positions_list = [ele[0] for ele in seq_positions_list]

    # adjust target_position
    target_position_list.sort(key=lambda data: data[1], reverse=True)
    new_target_position_list = [ele[0] for ele in target_position_list]


    return embedding_datas, new_target_embeddings, new_targets, data_length, new_seq_positions_list, new_target_position_list

if __name__ == "__main__":
    # data = SeqDataset(embedding_dir="/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/embeddings",
    #                   annotations="/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/train_folds8_seed300.pkl",
    #                   folds=[1, 2, 3, 4, 5, 6, 7])

    #############################################清洗只包含一个slice的患者########################################################
    with open("/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/train_folds8_seed300.pkl", 'rb') as f:
        df = pickle.load(f)
    reserved_index = []
    for index in tqdm(range(df.shape[0])):
        row = df.iloc[index]
        ID = row["ID"]
        PatientID = row["SeriesInstanceUID"]
        seq_frame = df[df["SeriesInstanceUID"] == PatientID].sort_values("PositionOrd")
        if seq_frame.shape[0] == 1:
            print("ok")
        else:
            reserved_index.append(index)

    df = df.iloc[reserved_index]
    print(df.shape)
    print(len(reserved_index))

    with open("/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/train_folds8_seed300_delete_one_slice_SID.pkl", 'wb') as f:
        pickle.dump(df, f)
