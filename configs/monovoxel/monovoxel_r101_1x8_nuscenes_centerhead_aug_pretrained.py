plugin = True
plugin_dir = 'det3d/'
#
_base_ = ['./monovoxel_r101_1x8_nuscenes_centerhead_aug.py']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [-49.92, -49.92, -2.92, 49.92, 49.92, 0.92]
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
model = dict(backbone=dict(
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False)))

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False)

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CustomMultiViewImageResize3D',
         img_scale=(1600, 900),
         ratio_range=(0.95, 1.05),
         keep_ratio=True,
         multiscale_mode='range'),
    # dict(type='CustomMultiViewImageCrop3D', crop_size=(900, 1600), rel_offset_h=(0.5, 1.)),
    dict(type='CustomMultiViewRandomFlip3D',
         flip_ratio=0.5,
         direction='horizontal'),
    dict(type='LoadAnnotations3D',
         with_bbox_3d=True,
         with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomMultiViewImagePad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    #dict(type='RandomScaleMultiviewImage3D', img_scale=(800, 450), keep_ratio=True, rescale_intrinsic=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomMultiViewImagePad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type='ConcatDataset',
             datasets=[
                 dict(type=dataset_type,
                      data_root=data_root,
                      ann_file=data_root + 'nuscenes_mini_infos_train.pkl',
                      pipeline=test_pipeline,
                      modality=input_modality,
                      classes=class_names,
                      test_mode=True,
                      box_type_3d='LiDAR'),
                 dict(type=dataset_type,
                      data_root=data_root,
                      ann_file=data_root + 'nuscenes_infos_val.pkl',
                      pipeline=test_pipeline,
                      modality=input_modality,
                      classes=class_names,
                      test_mode=True,
                      box_type_3d='LiDAR'),
             ],
             separate_eval=True),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_val.pkl',
              pipeline=test_pipeline,
              modality=input_modality,
              classes=class_names,
              test_mode=True,
              box_type_3d='LiDAR'),
)

load_from = '~/checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'
