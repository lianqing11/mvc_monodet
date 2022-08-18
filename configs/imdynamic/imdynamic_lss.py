model = dict(
    type='ImVoxelNetInteract',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # with_cp=True,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    # neck=dict(
    #     type='FlexibleFPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=64,
    #     num_outs=1),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_3d=dict(
        type='InteracitveNeckLSS',
        in_channels=64,
        out_channels=256,
        n_voxels=(156, 156, 12),
        voxel_size=(.64, .64, .32)),
    # neck_bev=dict(
    #     type='CustomNeckBEV',
    #     in_channels=256, 
    #     out_channels=256, 
    #     nlines=360, 
    #     npoints=234
    # ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            # _delete_=True,
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-49.6, -49.6, -1.80032795, 49.6, 49.6, -1.80032795],
                [-49.6, -49.6, -1.74440365, 49.6, 49.6, -1.74440365],
                [-49.6, -49.6, -1.68526504, 49.6, 49.6, -1.68526504],
                [-49.6, -49.6, -1.67339111, 49.6, 49.6, -1.67339111],
                [-49.6, -49.6, -1.61785072, 49.6, 49.6, -1.61785072],
                [-49.6, -49.6, -1.80984986, 49.6, 49.6, -1.80984986],
                [-49.6, -49.6, -1.763965, 49.6, 49.6, -1.763965],
            ],
            sizes=[
                [1.95017717, 4.60718145, 1.72270761],  # car
                [2.4560939, 6.73778078, 2.73004906],  # truck
                [2.87427237, 12.01320693, 3.81509561],  # trailer
                [0.60058911, 1.68452161, 1.27192197],  # bicycle
                [0.66344886, 0.7256437, 1.75748069],  # pedestrian
                [0.39694519, 0.40359262, 1.06232151],  # traffic_cone
                [2.49008838, 0.48578221, 0.98297065],  # barrier
            ],
            # custom_values=[0, 0],
            rotations=[0, 1.57],
            reshape_out=True),
        assigner_per_size=False,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        dir_limit_offset=0,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    n_voxels=(156, 156, 12),
    voxel_size=(.64, .64, .32))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        iou_calculator=dict(type='BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.3,
        min_pos_iou=0.3,
        ignore_iof_thr=-1),
    allowed_border=0,
    code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    use_rotate_nms=True,
    nms_across_levels=False,
    nms_pre=1000,
    nms_thr=0.2,
    score_thr=0.05,
    min_bbox_size=0,
    max_num=500)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'NuScenesMultiViewDataset'
data_root = '/share/jin.chen4/nuscenes_mm3d_sweep0/'
point_cloud_range = [-49.92, -49.92, -2.92, 49.92, 49.92, 0.92]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            # dict(type='Resize', 
            #     img_scale=[(1500, 843), (1700, 956)],
            #     keep_ratio=True,
            #     multiscale_mode='range'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])]
test_pipeline = [ 
    dict(
        type='MultiViewPipeline',
        n_images=6,
        transforms=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True), # generate a downsampel version
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32)]),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img'])]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    # train=dict(
    #     type='RepeatDataset',
    #     times=1,
    #     dataset=dict(
    #         type='ConcatDataset',
    #         datasets=[dict(
    #                 type=dataset_type,
    #                 data_root=data_root,
    #                 ann_file=data_root + 'nuscenes_infos_train.pkl',
    #                 pipeline=train_pipeline,
    #                 modality=input_modality,
    #                 classes=class_names,
    #                 box_type_3d='LiDAR',
    #                 test_mode=False),
    #             dict(
    #                 type=dataset_type,
    #                 data_root=data_root,
    #                 ann_file=data_root + 'nuscenes_infos_val.pkl',
    #                 pipeline=train_pipeline,
    #                 modality=input_modality,
    #                 classes=class_names,
    #                 box_type_3d='LiDAR',
    #                 test_mode=False),],)),
    val=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_val.pkl',
            pipeline=test_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=True,
            box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))




optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
# lr_config = dict(policy='step', step=[8, 11])
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)


total_epochs = 24

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook', init_kwargs=dict(project="det3d"))
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = 'work_dirs/fcos3d-new2.pth'
# resume_from = './work_dirs/imvoxelnet_nuscenes_mcls_large_aug/0411_2110-/latest.pth'
resume_from = None
workflow = [('train', 1)]




