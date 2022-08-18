plugin=True
plugin_dir='det3d/'

#
point_cloud_range=[2, -30.4, -3, 59.6, 30.4, 1]
# voxel_size=[0.05, 0.05, 0.1] # [0.2, 0.2, 0.2]
stereo_voxel_size=[0.2, 0.2, 0.2]
cv_dim=32

use_GN = True
model = dict(
    type='LigaStereo',
    maxdisp=288,
    downsampled_disp=4,
    point_cloud_range=point_cloud_range,
    voxel_size=stereo_voxel_size,
    img_feature_attentionbydisp=True,
    voxel_attentionbydisp=False,
    use_stereo_out_type="feature",
    backbone=dict(
        type='CustomResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        strides=[1, 2, 1, 1],
        dilations=[1, 1, 2, 4],
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet34'),
        num_channels_factor=[1, 2, 2, 2],
        with_max_pool=False,
        style='pytorch'),
    neck_stereo=dict(
        type="LigaStereoNeck",
        in_dims=[3, 64, 128, 128, 128],
        with_upconv=True,
        use_GN=use_GN),
    neck_det2d=dict(
        type="FPN",
        in_channels=[32],
        out_channels=64,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=5,),
    build_cost_volume=dict(
        type="BuildCostVolume",
        volume_types=[{"type":"concat", "downsample":4}],),
    neck_cost_volume=dict(
        type='LigaCostVolumeNeck',
        input_dim=64,
        cv_dim=cv_dim,
        use_GN=use_GN,
        num_hg=1),
    neck_voxel=dict(
      type="LigaVoxelNeck",
      num_3dconvs=1,
      input_dim=cv_dim*2, # since cat the image feature.
      rpn3d_dim=32,),

    depth_head=dict(
      type="LigaDepthHead",
      num_hg=1,
      cv_dim=cv_dim),
    neck_voxel_to_bev=dict(
        type='HeightCompression',
        num_bev_features=160), # 32 * 5,
    neck_bev=dict(
        type="HourglassBEVNeck",
        input_channels=160,
        num_channels=64,
        use_GN=use_GN,),
    det3d_head=dict(
        type='LigaDetHead',
        use_GN=use_GN,
        num_classes=3,
        in_channels=64,
        feat_channels=64,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [2, -30.4, -0.6, 59.6 - .32, 30.4, -0.6],
                [2, -30.4, -0.6, 59.6 - .32, 30.4, -0.6],
                [2, -30.4, -1.78, 59.6 - .32, 30.4, -1.78],],
            sizes=[[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
            rotations=[0, 1.57],
            reshape_out=True),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

dataset_type = 'CustomKittiDataset'
data_root = 'data/kitti/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=False, use_camera=True)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    #dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='CustomRandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='RandomScaleImage3D',
         img_scale=[(1173, 352), (1387, 416)],
        keep_ratio=True,
        multiscale_mode='range',
        rescale_intrinsic=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1280, 384), # debug if I can remove it.
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
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
            multiview_index=["image_2", "image_3"],
            modify_yaw_offset=0.,
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
                multiview_index=["image_2", "image_3"],
                classes=class_names,
                modify_yaw_offset=0.,
                test_mode=True),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + 'kitti_infos_val.pkl',
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=test_pipeline,
                modality=input_modality,
                multiview_index=["image_2", "image_3"],
                modify_yaw_offset=0.,
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
        multiview_index=["image_2", "image_3"],
        classes=class_names,
        modify_yaw_offset=0.,
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
