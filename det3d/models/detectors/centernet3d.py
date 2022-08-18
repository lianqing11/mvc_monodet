import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck


from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result
from det3d.models.utils.semi_utils import split_source_target
from det3d.models.utils import SSIM

#from mmdet3d.core.utils.mask import mask_background_region

import os.path as osp
from mmdet3d.core.bbox.iou_calculators import bbox_overlaps_3d
from mmdet3d.models.builder import build_loss
from mmdet3d.core.bbox.structures import Box3DMode

import torch.nn.functional as F
from det3d.core.bbox.util import alpha_to_ry, points_img2cam, points_img2cam_batch, \
                    projected_gravity_center, projected_2d_box, bbox_alpha

from mmcv import ops

@DETECTORS.register_module()
class CenterNet3D(BaseDetector):
    def __init__(self,
            backbone,
            neck,
            bbox_head,
            two_stage_infer=True,
            train_cfg=None,
            nocs_head=None,
            two_stage_head=None,
            test_cfg=None,
            pretrained=None,
            semi_bbox_in_supervised=False,
            loss_semi_bbox = dict(
                type="SemiBboxLoss"),
            roi_config = dict(
                type="RoIAlign", output_size=28,
                spatial_scale=1., sampling_ratio=0),
            match_bbox_mode = "iou",
            override_inference_max_num=None,
            stereo_post_process_head=None,
            init_cfg=None):
        '''
        Args:

        '''
        super().__init__(init_cfg)
        if pretrained:
            backbone.pretrained = pretrained

        self.backbone = build_backbone(backbone) # backbone from DLA 34

        self.neck = build_neck(neck) # identity module

        self.bbox_head = build_head(bbox_head) # centernet head
        self.pretrained=pretrained
        self.semi_bbox_in_supervised = semi_bbox_in_supervised
        self.match_bbox_mode = match_bbox_mode

        if nocs_head is not None:
            self.pred_nocs=True
            self.nocs_head = build_head(nocs_head)
        else:
            self.nocs_head = None
            self.pred_nocs = False


        self.two_stage_infer = two_stage_infer
        self.two_stage_head = build_head(two_stage_head) if two_stage_head is not None else None
        self.loss_semi_bbox = build_loss(loss_semi_bbox)
        roi_name = getattr(ops, roi_config['type'])
        roi_config.pop('type')
        self.roi_layer = roi_name(**roi_config)
        self.ssim_module = SSIM()
        self.override_inference_max_num = override_inference_max_num
        if stereo_post_process_head is not None:
            self.stereo_post_process_head = build_head(stereo_post_process_head)
        else:
            self.stereo_post_process_head = None
        # self.stereo_post_process = stereo_post_process

    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        N, V, C, H, W = img.shape
        # img = img.reshape([-1] + list(img.shape)[2:])
        img = img.reshape(-1, C, H, W)
        x = self.backbone(img)
        x = self.neck(x)
        if isinstance(x, tuple):
            x = [x[0]]

        # features_2d = self.head()
        #  = self.bbox_head.forward(x[-1], img_metas)
        return x

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        # get
        x = self.extract_feat(img, img_metas, "train")
        # if "dense_depth" in kwargs:
        #     dense_depth = kwargs["dense_depth"]
        # else:
        #     dense_depth = None
        losses, preds = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs)
        if self.pred_nocs:
            losses_ncos, preds_nocs = self.nocs_head.forward_train(
                x, img_metas, **kwargs)
            preds.update(preds_nocs)
            preds["normalize_nocs"] = self.nocs_head.normalize_nocs
            losses.update(losses_ncos)

        if self.two_stage_head is not None:
            preds["features"] = x
            gt_bboxes = kwargs["gt_bboxes"]
            bbox_list = self.bbox_head.get_bboxes(preds, img_metas)
            losses_two_stage, preds = self.two_stage_head.forward_train(
                                bbox_list, preds, img, img_metas,
                                gt_bboxes_3d, gt_labels_3d, gt_bboxes)
            losses.update(losses_two_stage)
        if self.semi_bbox_in_supervised:
            semi_bbox_loss = []
            target_img, target_img_metas = \
                    kwargs["target_img"], kwargs["target_img_metas"]

            source_bbox_results = self.bbox_head.get_bboxes(
                                                    preds, img_metas,
                                                    override_inference_max_num=\
                                                        self.override_inference_max_num)
            target_bbox_results = self.forward_test(
                                    target_img, target_img_metas,
                                    to_results=False,
                                    override_inference_max_num=\
                                        self.override_inference_max_num)

            for idx, (source_bbox_result, target_bbox_result) in \
                    enumerate(zip(source_bbox_results, target_bbox_results)):
                source_bbox, source_scores, source_label = source_bbox_result
                target_bbox, target_scores, target_label = target_bbox_result
                match_scores, max_inds = self.match_bbox(
                                                    source_bbox,
                                                    target_bbox,
                                                    img[idx:idx+1],
                                                    target_img[idx:idx+1],
                                                    img_metas[idx],
                                                    target_img_metas[idx])
                # overlap = bbox_overlaps_3d(source_bbox.tensor, target_bbox.tensor)
                # match_scores, max_inds = overlap.max(dim=1)
                semi_bbox_loss.append(
                    self.loss_semi_bbox(source_bbox,
                                        target_bbox[max_inds],
                                        match_scores,
                                        source_scores))
            # source_bbox, source_score, source_label = source_bbox_results
            if len(semi_bbox_loss) > 0:
                semi_bbox_loss = torch.stack(semi_bbox_loss).mean()
            else:
                semi_bbox_loss = img.new_zeros(1).mean()
                semi_bbox_loss.requires_grad=True
            losses["labeled_semi_bbox_loss"] = semi_bbox_loss
        return losses
    @torch.no_grad()
    def match_bbox(self,
                    source_bboxes,
                    target_bboxes,
                    img,
                    target_img,
                    img_meta,
                    target_img_meta):



        if self.match_bbox_mode == "iou":
            overlap = bbox_overlaps_3d(source_bboxes.tensor, target_bboxes.tensor)
            match_scores, max_inds = overlap.max(dim=1)
            return match_scores, max_inds
        elif self.match_bbox_mode == "ssim":
            with torch.no_grad():

                # convert to 2d box
                # do the roi align
                source_intrinsics = img_meta["cam2img"][0]
                source_extrinsics = img_meta["lidar2cam"][0]
                source_intrinsics = img.new_tensor(source_intrinsics)
                source_extrinsics = img.new_tensor(source_extrinsics)

                target_intrinsics = target_img_meta["cam2img"][0]
                target_extrinsics = target_img_meta["lidar2cam"][0]
                target_intrinsics = img.new_tensor(target_intrinsics)
                target_extrinsics = img.new_tensor(target_extrinsics)
                source_bbox_cam = source_bboxes.convert_to(
                            Box3DMode.CAM, source_extrinsics)
                source_bbox_2d = projected_2d_box(source_bbox_cam,
                            rt_mat=source_intrinsics,
                            img_shape=img_meta["img_shape"][0])

                source_bbox_2d = torch.cat(
                    [source_bbox_2d.new_ones(len(source_bbox_2d), 1)*0, source_bbox_2d],
                    dim=-1)
                source_features = self.roi_layer(img[:,0], source_bbox_2d.detach())

                target_bbox_cam = target_bboxes.convert_to(
                            Box3DMode.CAM, rt_mat=target_extrinsics)
                target_bbox_2d = projected_2d_box(target_bbox_cam,
                            rt_mat=target_intrinsics,
                            img_shape=target_img_meta["img_shape"][0])

                target_bbox_2d = torch.cat(
                    [target_bbox_2d.new_ones(len(target_bbox_2d), 1)*0, target_bbox_2d],
                    dim=-1)
                target_features = self.roi_layer(target_img[:,0], target_bbox_2d.detach())
                overlap = []
                for idx in range(len(source_features)):
                    source_feature_idx = source_features[idx:idx+1].expand(
                                                len(target_features), -1, -1, -1)
                    overlap_idx = self.ssim_module(source_feature_idx, target_features)
                    # print((source_feature_idx - target_features).abs().mean([1,2,3]))
                    overlap_idx = overlap_idx.mean(dim=[1,2,3]).reshape(1, -1)
                    overlap.append(overlap_idx)

                overlap = torch.cat(overlap, dim=0)
                print(overlap, bbox_overlaps_3d(source_bboxes.tensor, target_bboxes.tensor))
                # overlap = self.ssim_module(source_features, target_features)
                match_scores, max_inds = overlap.max(dim=1)
                return match_scores, max_inds


    def train_step(self, data, optimizer, semi_data=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        if semi_data is None:
            losses = self(**data)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

            return outputs
        else:
            data = split_source_target(data)
            losses = self(**data)
            semi_data = split_source_target(semi_data)
            semi_losses = self.forward_semi_train(**semi_data)
            losses.update(semi_losses)
            assert len(losses) > 0
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

            return outputs

    def forward_dummy(self, img):

        if img.dim() == 4: # fix the situation of multiview inputs
            img = img.unsqueeze(0)
        x = self.extract_feat(img, None, 'test')
        outs = self.bbox_head.forward(x)
        return outs

    def forward_semi_train(self, img, img_metas,
                                target_img, target_img_metas, **kwargs):
        losses = {}
        semi_bbox_loss = []
        source_bbox_results = self.forward_test(
                                img, img_metas, to_results=False,
                                override_inference_max_num=\
                                    self.override_inference_max_num)
        target_bbox_results = self.forward_test(
                                target_img, target_img_metas,
                                to_results=False,
                                override_inference_max_num=\
                                    self.override_inference_max_num)
        for source_bbox_result, target_bbox_result in \
                zip(source_bbox_results, target_bbox_results):
            source_bbox, source_scores, source_label = source_bbox_result
            target_bbox, target_scores, target_label = target_bbox_result
            match_scores, max_inds = self.match_bbox(
                                                source_bbox,
                                                target_bbox,
                                                img,
                                                target_img,
                                                img_metas,
                                                target_img_metas)
            semi_bbox_loss.append(
                self.loss_semi_bbox(source_bbox,
                                    target_bbox[max_inds],
                                    match_scores,
                                    source_scores))
        # source_bbox, source_score, source_label = source_bbox_results
        if len(semi_bbox_loss) > 0:
            semi_bbox_loss = torch.stack(semi_bbox_loss).mean()
        else:
            semi_bbox_loss = img.new_zeros(1).mean()
            semi_bbox_loss.requires_grad=True
        losses["unlabeled_semi_bbox_loss"] = semi_bbox_loss
        return losses

    def forward_test(self, img, img_metas, to_results=True,
                    override_inference_max_num=None,
                    **kwargs):

        if isinstance(img_metas[0]['ori_shape'], tuple):
            return self.simple_test(img, img_metas,
                                   to_results=to_results,
                                   override_inference_max_num=\
                                    override_inference_max_num,
                                   **kwargs)
        else:
            return self.aug_test(img, img_metas, **kwargs)

    def simple_test(self,
                    img,
                    img_metas,
                    to_results=True,
                    override_inference_max_num=None,
                    **kwargs):
        N, V, C, H, W = img.shape
        if self.stereo_post_process_head is not None:
            decoded_data = split_source_target(
                dict(img=img,
                     img_metas=img_metas))
            img, img_metas = decoded_data["img"], decoded_data["img_metas"]
            target_img, target_img_metas = \
                    decoded_data["target_img"], decoded_data["target_img_metas"]


        x = self.extract_feat(img, img_metas, "test")
        preds = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(preds, img_metas, img=img,
                            override_inference_max_num=override_inference_max_num)

        preds["features"] = x
        # if V == 2:
        #     right_x = self.extract_feat(right_img, img_metas, "test")
        #     right_preds = self.bbox_head(right_x)
        #     for key, item in right_preds.items():
        #         preds["right_" + key] = item
        #     preds["right_features"] = right_x
        #     kwargs["right_img"] = right_img

        if self.stereo_post_process_head is not None:
            bbox_list = self.stereo_post_process_head.forward_test(
                                                                    img,
                                                                    target_img,
                                                                    img_metas,
                                                                    target_img_metas,
                                                                    bbox_list,
                                                                    **kwargs)
        if to_results:
            bbox_results = [
                bbox3d2result(det_bboxes, det_scores, det_labels)
                for det_bboxes, det_scores, det_labels in bbox_list
            ]
            return bbox_results
        else:
            return bbox_list



    def aug_test(self, img, img_metas, rescale=False):
        # only assume batch size = 1
        feats = self.extract_feat(img, img_metas, 'test')

        # only support aug_test for one sample
        outs_list = self.bbox_head(feats[-1])
        for key, item in outs_list.items():
            if item is None:
                continue
            new_item = []
            for i in range(img.shape[1]):
                item_idx = item[i]
                if img_metas[0]['flip'][i] == True:
                    item_idx = torch.flip(item_idx, dims=[2])
                    if key == "offset":
                        item_idx[0] *= -1
                else:
                    continue
                new_item.append(item_idx)
            item = torch.stack(new_item, dim=0).mean(0, keepdim=True)
            outs_list[key] = item

        img_metas[0]['lidar2cam'] = img_metas[0]['lidar2cam'][0]
        bbox_list = self.bbox_head.get_bboxes(outs_list, img_metas)

        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results



    def show_results(self, *args, **kwargs):
        pass




