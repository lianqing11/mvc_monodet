plugin=True
plugin_dir='det3d/'
_base_ = ['../centernet3d_dla34_kitti.py']

eval_dataset_type = 'CustomKittiDataset'
model=dict(
    stereo_post_process_head = dict(
      type="StereoPostProcessModule"),
    bbox_head=dict(
      conv_channel=256,
      inference_max_num=10,)
    )
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
class_names = ['Car', 'Pedestrian', 'Cyclist']
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]

train_pipeline = [
    dict(type='LoadAnnotations3D',
         with_bbox=True,),
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomRandomFlip3Dv2', flip_ratio=0.5),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes'])]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadMultiviewSegMaskFromFile'),
    #dict(type='CustomRandomFlip3Dv2', flip_ratio=1.0),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='CustomDefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_multiview_semantic_seg'])
]

data = dict(
    test=dict(
      pipeline=test_pipeline,
      type=eval_dataset_type))
