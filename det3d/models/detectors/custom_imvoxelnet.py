import numpy as np
import torch
from mmdet3d.core import bbox3d2result, build_prior_generator
from mmdet3d.models.dense_heads import CenterHead
from mmdet3d.models.detectors import ImVoxelNet
from mmdet.models import DETECTORS
from mmdet3d.models.builder import build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from mmcv.runner import auto_fp16

from det3d.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class CustomImVoxelNet(BaseDetector):
    r"""' Modify ImVoxelNet to satisfy current object detection modules."""
    def __init__(self,
                 backbone,
                 neck,
                 neck_3d=None,
                 view_transform=None,
                 neck_bev=None,
                 bbox_head=None,
                 n_voxels=None,
                 anchor_generator=None,
                 select_first_neck_feat=True,
                 bev_det_format=False,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 use_grid_mask=False):
        super().__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.n_voxels = n_voxels
        self.bev_det_format = bev_det_format
        if anchor_generator is not None:
            self.anchor_generator = build_prior_generator(anchor_generator)
        else:
            self.anchor_generator = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.grid_mask = GridMask(True,
                                  True,
                                  rotate=1,
                                  offset=False,
                                  ratio=0.5,
                                  mode=1,
                                  prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.select_first_neck_feat = select_first_neck_feat
        if neck_3d is not None:
            self.neck_3d = build_neck(neck_3d)
        else:
            self.neck_3d = None
        if neck_bev is not None:
            self.neck_bev = build_neck(neck_bev)
        else:
            self.neck_bev = None
        if view_transform is not None:
            self.view_transform = build_neck(view_transform)
        else:
            self.view_transform = None

    def extract_img_feat(self, img, img_metas):
        B = len(img)

        input_shape = img.shape[-2:]

        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if img.dim() == 4:
            img = img.unsqueeze(0)
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.backbone(img)
        if self.select_first_neck_feat:
            x_fov = self.neck(img_feats)[0]
        else:
            x_fov = self.neck(img_feats)

        return x_fov

    

    def extract_feat(self, img, img_metas):
        """Extract 3d features from the backboen -> fpn -> 3d projection.

        Args:
            img (torch.Tensor): Input images of shape (B, N, C_in, H, W)
            img_metas (list): Image metas.

        Returns:
            torch.Tensor: of shape (B, C_out, N_x, N_y, N_z)
        """
        # modify the shape
        if self.bev_det_format is True:
            img, aug_config = img[0], img[1:]

        B = len(img)
        x_fov = self.extract_img_feat(img, img_metas)

        _, feat_C, feat_H, feat_W = x_fov.shape
        x_fov = x_fov.reshape(B, -1, feat_C, feat_H, feat_W)

        if self.anchor_generator is not None:
            points = self.anchor_generator.grid_anchors(
                [self.n_voxels[::-1]], device=x_fov.device)[0][:, :3]
        else:
            points = None
        if self.bev_det_format is False:
            x_3d = self.view_transform(x_fov, img_metas, points)
        else:
            x_3d = self.view_transform([x_fov] + aug_config)
        if self.neck_3d is not None:
            x_bev = self.neck_3d(x_3d)
        else:
            x_bev = x_3d
        if self.neck_bev is not None:
            x_bev = self.neck_bev(x_bev)

        return x_3d, x_bev, x_fov


    @auto_fp16(apply_to=('img', ))
    def forward(self, img_metas,
                      img=None,
                      img_inputs=None,
                      return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            return self.forward_train(img_metas,
                                      img=img,
                                      img_inputs=img_inputs,
                                      **kwargs)
        else:
            return self.forward_test(img_metas, img=img, img_inputs=img_inputs, **kwargs)
    def forward_test(self,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    **kwargs):
        # currently the model do not support test time augmentation
        if img is None:
            img = img_inputs
        if not isinstance(img, list):
            img = [img]
            img_metas = [img_metas]
        return self.simple_test(img[0], img_metas[0])

    # def simple_test(self, img_metas, img=None, rescale=False):
    #     pass
    def forward_train(self, img_metas,
                      img=None,
                      img_inputs=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      **kwargs):
        """Forward of training.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss components.
        """
        if img is None:
            img = img_inputs

        x_3d, x_bev, x_fov = self.extract_feat(img, img_metas)
        x = self.bbox_head(x_bev)
        if not isinstance(self.bbox_head, CenterHead):
            losses = self.bbox_head.loss(*x, gt_bboxes_3d, gt_labels_3d,
                                         img_metas)
        else:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, x]
            losses = self.bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, img, img_metas, get_feats=None):
        """Test without augmentations.

        Args:
            img (torch.Tensor): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        x_3d, x_bev, x_fov = self.extract_feat(img, img_metas)
        x = self.bbox_head(x_bev)

        if not isinstance(self.bbox_head, CenterHead):
            bbox_list = self.bbox_head.get_bboxes(*x, img_metas)
        else:
            bbox_list = self.bbox_head.get_bboxes(x, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        if get_feats is None or get_feats is False:
            return bbox_results
        elif get_feats == '3d':
            return bbox_results, x_3d
        elif get_feats == 'bev':
            return bbox_results, x_bev
        elif get_feats == 'fov':
            return bbox_results, x_fov


    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        Args:
            imgs (list[torch.Tensor]): Input images of shape (N, C_in, H, W).
            img_metas (list): Image metas.

        Returns:
            list[dict]: Predicted 3d boxes.
        """
        raise NotImplementedError
