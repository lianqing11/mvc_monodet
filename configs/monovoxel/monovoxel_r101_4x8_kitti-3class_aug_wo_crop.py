plugin=True
plugin_dir='det3d/'

##
_base_ = ['./monovoxel_r50_4x8_kitti-3class_aug_wo_crop.py']


model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
        style='pytorch'))