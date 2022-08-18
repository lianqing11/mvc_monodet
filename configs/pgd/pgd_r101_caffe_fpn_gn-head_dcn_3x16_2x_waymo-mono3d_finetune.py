_base_ = './pgd_r101_caffe_fpn_gn-head_dcn_3x16_2x_waymo-mono3d.py'
# model settings
model = dict(
    train_cfg=dict(code_weight=[
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0
    ]))
# optimizer
optimizer = dict(lr=0.004)
load_from = '/mnt/lustre/wangtai.vendor/mmdet3d-DfM/work_dirs/3x16-D3-kpts/latest.pth'  # noqa: E501
