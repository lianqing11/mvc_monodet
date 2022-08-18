import mmcv
import torch
import torch.nn as nn

from mmdet.models.builder import LOSSES
# from .utils import weighted_loss
import torch.nn.functional as F

@LOSSES.register_module()
class UncertaintyL1Loss(nn.Module):
    """L1 loss with modeling laplacian distribution

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean',
                       loss_weight=1.0, 
                       uncertainty_weight=1.0,
                       uncertainty_range=[-10, 10]):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.uncertainty_weight = uncertainty_weight
        self.uncertainty_range = uncertainty_range
    
    def forward(self, pred, target, uncertainty,
                     weight=None, avg_factor=None,
                     reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss = F.l1_loss(
            pred, target, weight, reduction="none",)
        uncertainty_clamp = uncertainty.clamp(-10, 10)
        loss = loss * torch.exp(- uncertainty_clamp) + \
                        uncertainty_clamp * uncertainty_weight
        loss = self.loss_weight * loss.mean()
        return loss

@LOSSES.register_module()
class SemiBboxLoss(nn.Module):
    """
    Loss for cross-view consistency in the bounding box level.
    Args:
        reduction (str, optinal): The method to reduce the loss.
            Optional are "none", "mean" and "sum".
        loss_weight (float, optinal): The weight of loss.
    """

    def __init__(self, reduction="mean",
                    loss_weight=0.1,
                    loss_type = "corner_loss", # corner or bbox loss
                    loss_mode = "l1", # l1 or weighted_l1,
                    conf_threshold=0.5,
                    match_threshold=0.2):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.loss_mode = loss_mode
        self.conf_threshold = conf_threshold
        self.match_threshold = match_threshold

    def forward(self, source_bbox,
                      target_bbox,
                      match_scores, 
                      scores, avg_factor=None):
        """
        source_bbox (bbox instances): The boudning box from source img.
        target_bbox (bbox instances): The bounding box from target img.
        mask (tensor): the iou / ssim criterion with shape of Nx1.
        scores (tensor): the output conf threshold.
        avg_factor (bool): 
        """
        assert self.reduction == "mean"
        mask = (match_scores > self.match_threshold) & \
                (scores > self.conf_threshold)
        if mask.sum() > 0:
            if self.loss_type == "corner_loss":
                source_tensor = source_bbox.corners[mask.detach()]
                target_tensor = target_bbox.corners[mask.detach()]
                loss = F.l1_loss(source_tensor, target_tensor, reduction="none")
                loss = loss.mean(dim=[1,2])
            else:
                source_tensor = source_bbox.tensor[mask.detach()]
                target_tensor = target_bbox.tensor[mask.detach()]
                loss = F.l1_loss(source_tensor, target_tensor, reduction="none")

                loss = loss.sum(dim=1)
            
            if self.loss_mode == "weighted_l1":
                loss = loss * scores[mask]
            loss = self.loss_weight * loss
            return loss.mean()
        else:
            loss = source_bbox.tensor.new_zeros(1).mean()
            loss.requres_grad=True
            return loss

