
import torch
from mmdet.models import build_detector

from mmtrack.core import outs2results, results2outs
from mmdet.models import DETECTORS # use detectors for registory.
from mmtrack.models.builder import build_motion, build_tracker
from mmtrack.models.mot.base import BaseMultiObjectTracker

@DETECTORS.register_module()
class OCSORT(BaseMultiObjectTracker):
    """OCSORT in 3D task.
    """

    def __init__(self,
                detector=None,
                tracker=None,
                motion=None,
                get_feats='bev',
                init_cfg=None,
                pretrained=None,
                train_cfg=None,
                test_cfg=None,):
        
        super().__init__(init_cfg)

        if detector is not None:
            self.detector = build_detector(detector)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.get_feats = get_feats


    
    def forward_train(self, *args, **kwargs):
            """ Forward function during training"""
            return self.detector.forward_train(*args, **kwargs)

        
    def forward_test(self, img, img_metas, **kwargs):
        # currently the model do not support test time augmentation
        if not isinstance(img, list):
            img = [img]
            img_metas = [img_metas]
        return self.simple_test(img[0], img_metas[0])


    def simple_test(self, img, img_metas, rescale=False, **kwargs):
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
        det_results = self.detector.simple_test(
            img, img_metas)
        det_results = det_results[0]
        # det_bboxes = torch.tensor
        bboxes = det_results['bboxes_3d']
        scores = det_results['scores_3d']
        labels = det_results['labels_3d']

        track_bboxes, track_scores, track_labels, track_ids = self.tracker.track(
            img_metas=img_metas,
            model=self,
            bboxes=bboxes,
            confidence=scores,
            labels=labels, 
            frame_id=self.frame_id,
            rescale=rescale,
            **kwargs)
        
        return dict(
            bboxes_3d=track_bboxes,
            scores_3d=track_scores,
            labels_3d=track_labels,
            ids=track_ids)
        

