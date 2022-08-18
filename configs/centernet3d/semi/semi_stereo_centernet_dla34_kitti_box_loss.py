plugin=True
plugin_dir='det3d/'

#
model = dict(
    match_bbox_mode="ssim",
    semi_bbox_in_supervised=True,
    type='CenterNet3D',
    backbone=dict(
        type='DCNDLA',),
    neck=dict(
        type="IdentityNeck",),
    bbox_head=dict(
        type='CenterNet3DHead',
        num_classes=3,
        input_channel=64,
        conv_channel=256,),
    override_inference_max_num=10,
    loss_semi_bbox = dict(
        type="SemiBboxLoss",
        match_threshold=0.2))
)



img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'CustomKittiDataset'
semi_dataset_type = 'CustomKittiDataset'
eval_dataset_type = "CustomMonoKittiDataset"
data_root = 'data/kitti/'
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

train_cfg = dict()
test_cfg = dict()


train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_bbox=True,),
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomRandomFlip3Dv3', flip_ratio=0.5),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes'])]

train_semi_pipeline = [
    # dict(type='LoadAnnotations3D',
        #  with_bbox=True,),
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomRandomFlip3Dv3', flip_ratio=0.5),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img'])]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]
post_semi_pipeline = [
    dict(type="SplitSourceTarget", source_index=0)
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            remove_hard_instance_level=1,
            # post_semi_pipeline=post_semi_pipeline,
            test_mode=False)),
    semi_train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=semi_dataset_type,
            data_root=data_root,
            ann_file=data_root + 'kitti_infos_raw_clean.pkl',
            split='kitti_raw',
            pts_prefix='velodyne_reduced',
            pipeline=train_semi_pipeline,
            modality=input_modality,
            classes=class_names,
            unlabeled_split=True,
            # post_semi_pipeline=post_semi_pipeline,
            test_mode=True)),

    val=dict(
        type='ConcatDataset',
        datasets=[dict(
                type=eval_dataset_type,
                data_root=data_root,
                ann_file=data_root + 'kitti_infos_train.pkl',
                split='training',
                pts_prefix='velodyne_reduced',
                pipeline=test_pipeline,
                modality=input_modality,
                classes=class_names,
                test_mode=True),
            dict(
                type=eval_dataset_type,
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
        type=eval_dataset_type,
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
    lr=0.0003,
    weight_decay=0.00001,)
optimizer_config = dict(grad_clip=dict(max_norm=35., norm_type=2))
lr_config = dict(policy='step', step=[5, 8])
total_epochs = 10

checkpoint_config = dict(interval=50, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook', init_kwargs=dict(project="det3d"))
    ])
evaluation = dict(interval=1)
dist_params = dict(backend='nccl')
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = 'INFO'
load_from = "ckpts/centernet3d_dla34_kitti_conv256_filter_hard_0803_1952_epoch16.pth"
resume_from = None
workflow = [('train', 1)]
runner = dict(type='SemiEpochBasedRunner', max_epochs=total_epochs)
