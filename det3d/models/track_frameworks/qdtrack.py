import torch
from mmdet.models import build_detector, build_head
from mmtrack.core import outs2results, results2outs
from mmtrack.models.builder import build_tracker
from mmtrack.models.mot import BaseMultiObjectTracker, QDTrack
from mmdet.models import DETECTORS # use detectors for registory.


@DETECTORS.register_module()
class CustomQDTrack(QDTrack):
    def __init__(self,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 freeze_detector=False,
                 get_feats='bev',
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args,
                 **kwargs):
        BaseMultiObjectTracker.__init__(self, *args, **kwargs)
        self.pretrained = pretrained
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if detector is not None:
            self.detector = build_detector(detector)

        if track_head is not None:
            self.track_head = build_head(track_head)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')

        self.get_feats = get_feats
        self.frame_id = -1

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img,
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      **kwargs):
        """Forward function during training.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : Tensor]: All losses.
        """
        pass

    def forward_test(self, img, img_metas, **kwargs):
        # currently the model do not support test time augmentation
        if not isinstance(img, list):
            img = [img]
            img_metas = [img_metas]
        return self.simple_test(img[0], img_metas[0])

    def simple_test(self, img, img_metas, rescale=False):
        """Test forward.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): whether to rescale the bboxes.

        Returns:
            dict[str : Tensor]: Track results.
        """
        is_first_frame = img_metas[0].get('is_first_frame', -1)

        if is_first_frame == True:
            self.tracker.reset()
            self.frame_id = -1
        else:
            self.frame_id += 1

        bbox_results, feats = self.detector.simple_test(
            img, img_metas, self.get_feats)
        bbox_results = bbox_results[0]
        # det_bboxes = torch.tensor
        bboxes = bbox_results['bboxes_3d']
        scores = bbox_results['scores_3d']
        labels = bbox_results['labels_3d']

        # TODO check how to extract features and handle the data conversion issue.

        bboxes, scores, labels, ids = self.tracker.track(img_metas,
                                           feats=feats,
                                           model=self,
                                           bboxes=bbox_results,
                                           confidence=scores,
                                           labels=labels,
                                           frame_id=self.frame_id)
        return [dict(bboxes_3d=bboxes, scores_3d=scores, labels_3d=labels, ids=ids)]

