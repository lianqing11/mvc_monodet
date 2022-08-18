_base_ = ['./imvoxelnet_4x8_kitti-3d-car.py']

dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='LoadImageFromFile'),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Resize',
        img_scale=[(1173, 352), (1387, 416)],
        keep_ratio=True,
        multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1280, 384), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False)),
    val=dict(
        type='ConcatDataset',
        datasets=[dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'kitti_infos_train.pkl',
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'kitti_infos_val.pkl',
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True),],
        separate_eval=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True))

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
lr_config = dict(policy='step', step=[8, 11])
total_epochs = 12

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook'),
           dict(type='WandbLoggerHook', init_kwargs=dict(project="det3d"))])

evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # only 1 of 4 FPN outputs is used
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
