model = dict(
        type='ImVoxelNet',
        pretrained='torchvision://resnet101',
    backbone=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True)
    ),
    neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=64,
                num_outs=4
    ),
    neck_3d=dict(
                type='NuScenesImVoxelNeck',
                in_channels=64,
                out_channels=256
    ),
    bbox_head=dict(
                type='CenterHead',
                in_channels=256,
        tasks=[
                        dict(num_class=1, class_names=['car']),
                        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
                        dict(num_class=2, class_names=['bus', 'trailer']),
                        dict(num_class=1, class_names=['barrier']),
                        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
                        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),

        ],
        common_heads=dict(
                        reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
                share_conv_channel=64,
        bbox_coder=dict(
                        type='CenterPointBBoxCoder',
                        # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                        # pc_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                        pc_range=[-49.92, -49.92, -2.92, 49.92, 49.92, 0.92],
                        max_num=500,
                        score_threshold=0.1,
                        out_size_factor=1,
                        voxel_size=(.64, .64),
                        code_size=9
        ),
        seperate_head=dict(
                        type='SeparateHead', init_bias=-2.19, final_kernel=3
        ),
                loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
                norm_bbox=True
    ),
    n_voxels=(312, 312, 12),
        voxel_size=(.32, .32, .32)
)
# model training and testing settings
train_cfg = dict(
            grid_size=[156, 156, 1],
            voxel_size=(.64, .64, 3.84),
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range = [-49.92, -49.92, -2.92, 49.92, 49.92, 0.92]

)
test_cfg = dict(
            # post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_limit_range=[-49.92, -49.92, -2.92, 49.92, 49.92, 0.92],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=(.64, .64),
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2
)
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
        use_external=False
)

train_pipeline = [
        dict(type='LoadAnnotations3D',),
    dict(
                type='MultiViewPipeline',
                n_images=6,
        transforms=[
                        dict(type='LoadImageFromFile', to_float32=True),
                        # dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                                type='PhotoMetricDistortion',
                                brightness_delta=32,
                                contrast_range=(0.5, 1.5),
                                saturation_range=(0.5, 1.5),
                                hue_delta=18
            ),
                        # dict(type='Resize3D', img_scale=(1600, 900), keep_ratio=True, ratio_range=(0.8, 1.2), resize_depth=True),
                        dict(type='Resize', img_scale=(800, 450), keep_ratio=True),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='Pad', size_divisor=32)
        ]
    ),
        # dict(type='FovRandomFlip'),
        dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
        dict(type='DefaultFormatBundle3D', class_names=class_names),
        dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
                type='MultiViewPipeline',
                n_images=6,
        transforms=[
                        dict(type='LoadImageFromFile'),
                        dict(type='Resize', img_scale=(800, 450), keep_ratio=True), # generate a downsampel version
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='Pad', size_divisor=32)
        ]
    ),
        dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
        dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
        dict(type='Collect3D', keys=['img'])
]

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
                        box_type_3d='LiDAR'
        )
    ),
    val=dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=data_root + 'nuscenes_infos_train.pkl',
                    pipeline=test_pipeline,
                    classes=class_names,
                    modality=input_modality,
                    test_mode=True,
                    box_type_3d='LiDAR'
    ),
    test=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'nuscenes_infos_train.pkl',
                pipeline=test_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=True,
                box_type_3d='LiDAR'
    )
)




optimizer = dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
    paramwise_cfg=dict(
                custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}
    )
)
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
# lr_config = dict(policy='step', step=[8, 11])
lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        min_lr_ratio=1e-3
)


total_epochs = 24

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
        interval=10,
    hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook'),
                # dict(type='WandbLoggerHook', init_kwargs=dict(project="det3d"))

    ]
)
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = 'work_dirs/fcos3d-new2.pth'
resume_from = None
workflow = [('train', 1)]
