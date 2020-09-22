# -*- coding: utf-8 -*-
# @Time    : 2020/4/23 21:33
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : din.py
import glob
import pickle
import numpy as np
from tqdm import tqdm

path = "../../cache/embeddings"

sum_embedding_path = glob.glob(path+"/*.npy")

# for embedding_path in sum_embedding_path:
#     embedding = np.load(embedding_path)
#     print()

with open("/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/train_folds8_seed300.pkl", 'rb') as f:
    df = pickle.load(f)

grouped = df.sort_values('PositionOrd').groupby('PatientID')
for _,group in tqdm(grouped, total=len(grouped)):
    fold = group["fold"].nunique()
    assert fold == 1