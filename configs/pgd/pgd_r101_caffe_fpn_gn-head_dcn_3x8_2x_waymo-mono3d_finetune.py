_base_ = './pgd_r101_caffe_fpn_gn-head_dcn_3x8_2x_waymo-mono3d.py'
# model settings
model = dict(
    train_cfg=dict(code_weight=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]))
# optimizer
optimizer = dict(lr=0.002)
load_from = '/mnt/lustre/wangtai.vendor/mmdet3d-DfM/work_dirs/fcos3d-waymo-3x8-D3/latest.pth'  # noqa: E501
