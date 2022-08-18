import numpy as np
import torch
import os
from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
from tqdm import tqdm
from mmdet3d.datasets.pipelines import Compose
import os.path as osp
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)

@DATASETS.register_module()
class CustomKittiDataset(KittiDataset):
    r"""Kitti Dataset.
    This dataset adds camera intrinsics and extrinsics to the results.
    and support stereo data.

    multiview_index: index for selecting the images in the multivew database.
    """
    PathMapping = {
        "image_2": "image_02",
        "image_3": "image_03"}

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Camera',
                 filter_empty_gt=True,
                 test_mode=False,
                 modify_yaw_offset=0,
                 load_prev_frame=False,
                 remove_hard_instance_level=0,
                 unlabeled_split=False,
                 multiview_index = ["image_2", "image_3"],
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                 post_semi_pipeline=None):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        if post_semi_pipeline is not None:
            self.post_semi_pipeline = Compose(post_semi_pipeline)
        else:
            self.post_semi_pipeline = None
        self.remove_hard_instance_level = remove_hard_instance_level
        self.multiview_index = multiview_index
        self.default_multiview_index = "image_2"

        self.modify_yaw_offset = modify_yaw_offset

        self.load_prev_frame = load_prev_frame

        self.n_views = len(multiview_index)
        self.unlabeled_split = unlabeled_split
        if self.unlabeled_split:
            self._set_group_flag()
        if remove_hard_instance_level > 0:
            for idx, info in tqdm(enumerate(self.data_infos)):
                info['annos'] = self.remove_hard_instances(info['annos'])
                self.data_infos[idx] = info



    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root,
                                    info['image']['image_path'])

        # TODO: consider use torch.Tensor only

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)
        lidar2cam = rect @ Trv2c

        images_paths = []
        seg_mask_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam2img = []
        # lidar2img = P2 @ rect @ Trv2c
        for multiview_index_idx in self.multiview_index:
            temp_img_filename = img_filename.replace(
                        self.default_multiview_index, multiview_index_idx)
            temp_img_filename = temp_img_filename.replace(
                self.PathMapping[self.default_multiview_index],
                self.PathMapping[multiview_index_idx])

            images_paths.append(temp_img_filename)
            temp_seg_mask_filename = temp_img_filename.replace(
                "image_", "seg_mask_")
            temp_seg_mask_filename = temp_seg_mask_filename.replace(
                'png', 'npz')
            seg_mask_paths.append(temp_seg_mask_filename)
            if multiview_index_idx == "image_2":
                cam2img_idx = info['calib']['P2'].astype(np.float32)
            else:
                cam2img_idx = info['calib']['P3'].astype(np.float32)

            cam2img.append(cam2img_idx)

            lidar2img_rts.append(cam2img_idx @ lidar2cam)
            lidar2cam_rts.append(lidar2cam)
        if isinstance(sample_idx, int):
            pts_filename = self._get_pts_filename(sample_idx)
        else:
            pts_filename = osp.join(self.root_split,
                 info['point_cloud']['velodyne_path'])
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None,
            img_info=dict(filename=images_paths),
            img_filename=images_paths,
            seg_mask_filename=seg_mask_paths,
            lidar2img=lidar2img_rts,
            lidar2cam=lidar2cam_rts,
            cam2img=cam2img,)

        if not self.test_mode and 'annos' in info:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict


    def convert_valid_bboxes(self, box_dict, info):
        """
        Add the offset of yaw angle
        """
        if self.modify_yaw_offset!=0:
            box_dict['boxes_3d'].tensor[:, 6] += self.modify_yaw_offset

        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']

        sample_idx = info['image']['image_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)
        if isinstance(box_preds, CameraInstance3DBoxes):
            box_preds_camera = box_preds
            box_preds = box_preds_camera.convert_to(Box3DMode.LIDAR,
                                                    np.linalg.inv(rect @ Trv2c))
        else:
            box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )


    def remove_hard_instances(self, ann_info):
        if self.remove_hard_instance_level == 0:
            return ann_info
        elif self.remove_hard_instance_level == 1:
            # occluded >= 2
            # depth >= 60
            mask = (ann_info['occluded'] <=2) & \
                     (ann_info['location'][:, 2] <=60)
            for key, item in ann_info.items():
                ann_info[key] = item[mask]
        elif self.remove_hard_instance_level == 2:
            # 1.
            mask = (ann_info["location"][:, 2] < 80)
            mask = mask & (ann_info["location"][:, 2] > 0)
            mask = mask & (np.abs(ann_info["location"][:, 1]) < 40)
            truncated_mask = ann_info['truncated'] >=0.9
            truncated_mask = truncated_mask & \
                ((ann_info["bbox"][:,2:] - ann_info["bbox"][:,:2]).min(axis=1) <=20)

            mask = mask &  (~truncated_mask)
            # mask = mask & (ann_info['truncated'])
            for key, item in ann_info.items():
                ann_info[key] = item[mask]
        elif self.remove_hard_instance_level == 3:
            # 1.
            mask = (ann_info["location"][:, 2] < 60)
            mask = mask & (ann_info["location"][:, 2] > 0)
            mask = mask & (np.abs(ann_info["location"][:, 1]) < 40)
            truncated_mask = ann_info['truncated'] >=0.9
            truncated_mask = truncated_mask & \
                ((ann_info["bbox"][:,2:] - ann_info["bbox"][:,:2]).min(axis=1) <=20)

            mask = mask &  (~truncated_mask)
            # mask = mask & (ann_info['truncated'])
            for key, item in ann_info.items():
                ann_info[key] = item[mask]

        return ann_info




@DATASETS.register_module()
class CustomMonoKittiDataset(CustomKittiDataset):
    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='Camera',
                 filter_empty_gt=True,
                 test_mode=False,
                 modify_yaw_offset=0,
                 load_prev_frame=False,
                 remove_hard_instance_level=0,
                 unlabeled_split=False,
                 multiview_index = ["image_2"],
                 pcd_limit_range=[0, -40, -3, 70.4, 40, 0.0],
                 post_semi_pipeline=None):
        # if augmentation with stereo image,
        # the multiview index can be ["image_2", "image_3"]
        self.n_views = len(multiview_index)

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            modify_yaw_offset=modify_yaw_offset,
            remove_hard_instance_level=remove_hard_instance_level,
            load_prev_frame=load_prev_frame,
            unlabeled_split = unlabeled_split,
            multiview_index = multiview_index,
            pcd_limit_range=pcd_limit_range,
            post_semi_pipeline=post_semi_pipeline)
    def __len__(self):
        return self.n_views * super().__len__()

    def get_data_info(self, index):

        # handle the duplicate situation;
        # split the multiview image into single view image
        view_index = index % self.n_views
        index = index // self.n_views
        input_dict = super().get_data_info(index)
        input_dict['img_filename'] = [input_dict['img_filename'][view_index]]
        input_dict['lidar2img'] = [input_dict['lidar2img'][view_index]]
        input_dict['lidar2cam'] = [input_dict['lidar2cam'][view_index]]
        input_dict['cam2img'] = [input_dict['cam2img'][view_index]]
        if 'seg_mask_filename' in input_dict:
            input_dict['seg_mask_filename'] = [input_dict['seg_mask_filename'][view_index]]
        return input_dict


