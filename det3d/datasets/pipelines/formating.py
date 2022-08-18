from mmdet.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import Collect3D, DefaultFormatBundle3D
import numpy as np
@PIPELINES.register_module()
class CustomCollect3D(Collect3D):
    """
        Add lidar2cam, intrinsics to the meta keys
    
    """
    
    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'lidar2cam', 'cam2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug', 'ori_lidar2img', 'img_crop_offset',
                   'img_resized_shape', 'num_ref_frames', 'num_views', 
                   'is_first_frame', 'scene_token', 
                   'ego2global', 'transformation_3d_flow')):
        
        super().__init__(keys, meta_keys)

    

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Support multiview semantic seg mask"""

    def __call__(self, results):
        results = super(DefaultFormatBundle3D, self).__call__(results)
    
        if 'gt_multiview_semantic_seg' in results:
            results['gt_multiview_semantic_seg'] = \
                np.stack(results['gt_multiview_semantic_seg'], axis=0)
        return results