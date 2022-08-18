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


data = dict(
    test=dict(
      type=eval_dataset_type))
