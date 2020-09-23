import os
import sys
# name = os.path.basename(sys.argv[2]).split(".")[0]
FatherDir = sys.argv[2].split("/")[-2]
ChildDir = sys.argv[2].split("/")[-1].split(".")[0]
workdir = '/home/amax/xiangxi-ubuntu/zpf/target_work/MICCAI/model/{}/{}'.format(FatherDir, ChildDir)
annotations = '/home/amax/xiangxi-ubuntu/zpf/target_work/MICCAI/cache/train_folds8_seed300.pkl'
imgdir = '/home/amax/xiangxi-ubuntu/zpf/small_data/train_images'

auto=False
soft_window=False
seed = 2020

n_fold = 8
epoch = 10
resume_from = None

batch_size = 24
num_workers = 4
imgsize = (480, 480)

loss = dict(
    name='BCEWithLogitsLoss',
    params=dict(),
)

optim = dict(
    name='Adam',
    params=dict(
        lr=1.4e-4,
    ),
)

model = dict(
    name='resnet50',
    pretrained='imagenet',
    n_output=6,
)

scheduler = dict(
    name='MultiStepLR',
    params=dict(
        milestones=[3,5,7],
        gamma=3/7,
    ),
)


normalize = None
#normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],}
#normalize = {'mean': [13.197, 7.179, -78.954,], 'std': [24.509, 55.063, 113.127,]}


crop = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.7,1.0), p=1.0))
crop_test = dict(name='RandomResizedCrop', params=dict(height=imgsize[0], width=imgsize[1], scale=(0.75,1.0), p=1.0))
resize = dict(name='Resize', params=dict(height=imgsize[0], width=imgsize[1]))
hflip = dict(name='HorizontalFlip', params=dict(p=0.5,))
vflip = dict(name='VerticalFlip', params=dict(p=0.5,))
contrast = dict(name='RandomBrightnessContrast', params=dict(brightness_limit=0.08, contrast_limit=0.08, p=0.5))
totensor = dict(name='ToTensor', params=dict(normalize=normalize))
rotate = dict(name='Rotate', params=dict(limit=30, border_mode=0), p=0.7)
rotate_test = dict(name='Rotate', params=dict(limit=25, border_mode=0), p=0.7)
dicomnoise = dict(name='RandomDicomNoise', params=dict(limit_ratio=0.05, p=0.9))
dicomnoise_test = dict(name='RandomDicomNoise', params=dict(limit_ratio=0.04, p=0.7))

window_policy = 4

data = dict(
    train=dict(
        dataset_type='CustomDataset',
        annotations=annotations,
        imgdir=imgdir,
        imgsize=imgsize,
        n_grad_acc=1,
        loader=dict(
            shuffle=True,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[crop, hflip, rotate, dicomnoise, totensor],
        dataset_policy=1,
        window_policy=window_policy,
        soft_window=soft_window
    ),
    valid = dict(
        dataset_type='CustomDataset',
        annotations=annotations,
        imgdir=imgdir,
        imgsize=imgsize,
        loader=dict(
            shuffle=False,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        transforms=[totensor],
        dataset_policy=1,
        window_policy=window_policy,
        soft_window=soft_window
    )
)

###valid
n_tta = 1
snapshot = workdir + "/fold%d_ep%d.pt" % (0, epoch-1)