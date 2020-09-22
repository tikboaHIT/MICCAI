import os
import sys

name = os.path.basename(sys.argv[2]).split(".")[0]
workdir = '/home/amax/xiangxi-ubuntu/zpf/MICCAI/model/{}'.format(name)
embedding_dir="/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/embeddings"
annotations="/home/amax/xiangxi-ubuntu/zpf/MICCAI/cache/train_folds8_seed300_delete_one_slice_SID.pkl"

seed = 2020
n_fold = 8
epoch = 8
resume_from = None

batch_size = 36
num_workers = 4

loss = dict(
    name='BCEWithLogitsLoss',
    params=dict(),
)

optim = dict(
    name='Adam',
    params=dict(
        lr=2e-3,
    ),
)

model = dict(
    name='sequence',
    input_size=2048,
    hidden_size=1024,
    num_layers=2,
    n_output=6,
)

scheduler = dict(
    name='MultiStepLR',
    params=dict(
        milestones=[1,2,3],
        gamma=1/2,
    ),
)

data = dict(
    train=dict(
        dataset_type='CustomDataset',
        annotations=annotations,
        embedding_dir=embedding_dir,
        n_grad_acc=1,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        dataset_policy=1,
    ),
    valid = dict(
        dataset_type='CustomDataset',
        annotations=annotations,
        embedding_dir=embedding_dir,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        dataset_policy=1,
    )
)

###valid
n_tta = 1
snapshot = workdir + "/fold%d_ep%d.pt" % (0, epoch-1)