import torch
from torch import nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    """
    pred : [B*N, 4]
    box_gt : [B*N, 4]
    cls_gt : [B*N, 1]
    """
    def __init__(self, eps = 1e-5, **kwargs):

        super(L1Loss, self).__init__(**kwargs)
        self._eps = eps
        
        
    def forward(self, box_preds, box_targets, cls_targets, weight=None):

        box_preds = box_preds.reshape(-1,4)
        box_targets = box_targets.reshape(-1,4)
        cls_targets = cls_targets.reshape(-1,1)
        
#         ctr_targets = weight
#         ctr_targets = ctr_targets.reshape(-1, 1)
#         ctr_targets = ctr_targets[cls_targets>0]
#         weight = ctr_targets  
        
        mask = (cls_targets>0).expand_as(box_targets).view(-1,4)
        box_preds = box_preds[mask].view(-1,4)
        box_targets = box_targets[mask].view(-1,4).log()
#         print('box_preds',box_preds)
#         print('box_targets',box_targets)
#         print('l1loss',F.smooth_l1_loss(box_preds, box_targets, size_average=True))   
        losses = F.smooth_l1_loss(box_preds, box_targets, size_average=True)
#         print('losses',losses/box_preds.size(0))
        
#         print('iou2', iou)
#         print('iou2_loss', losses)
#         print('area_intersect', area_intersect)
#         print('area_union', area_union)
#         print('iou', iou)
#         print('weight', weight)
#         print('before weight losses', losses)

        if weight is not None:            
            losses = losses * weight    
#         print('after weight losses', losses)
        assert losses.numel() != 0
        return losses.mean()

        
class IOULoss(nn.Module):
    """
    pred : [B*N, 4]
    box_gt : [B*N, 4]
    cls_gt : [B*N, 1]
    """
    def __init__(self, eps = 1e-5, **kwargs):

        super(IOULoss, self).__init__(**kwargs)
        self._eps = eps
        
        
    def forward(self, box_preds, box_targets, cls_targets, weight=None):
#         pred_left = pred[:, :, 0]
#         pred_top = pred[:, :, 1]
#         pred_right = pred[:, :, 2]
#         pred_bottom = pred[:, :, 3]

#         target_left = target[:, :, 0]
#         target_top = target[:, :, 1]
#         target_right = target[:, :, 2]
#         target_bottom = target[:, :, 3]
        box_preds = box_preds.reshape(-1,4)
        box_targets = box_targets.reshape(-1,4)
        cls_targets = cls_targets.reshape(-1,1)
        if weight is not None:
            ctr_targets = weight
            ctr_targets = ctr_targets.reshape(-1, 1)
            ctr_targets = ctr_targets[cls_targets>0]
            weight = ctr_targets
            
        mask = (cls_targets>0).expand_as(box_targets).view(-1,4)
        box_preds = box_preds[mask].view(-1,4)
        box_targets = box_targets[mask].view(-1,4)

        
#         print('ctr_targets', ctr_targets.size())
#         print('ctr_targets', ctr_targets)
        
#         print('weight.sum()',weight.sum())
#         print('mask',mask.size())
#         print('box_preds',box_preds.size())
#         mask = (cls_targets>0) 
#         print('mask',mask.size())
        
        pred_left = box_preds[:, 0]
        pred_top = box_preds[:, 1]
        pred_right = box_preds[:, 2]
        pred_bottom = box_preds[:, 3]

        target_left = box_targets[:, 0]
        target_top = box_targets[:, 1]
        target_right = box_targets[:, 2]
        target_bottom = box_targets[:, 3]
        
        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
#         print('target',box_targets.size())
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)
                
        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
#         print('w_intersect.size()',w_intersect)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)
#         print('h_intersect.size()',h_intersect)
        
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        
#        print(area_intersect.size())
        iou = (area_intersect / area_union)
#         print('iou', iou)
#         print('iou1_loss',-torch.log(iou + self._eps))
#         iou = iou[torch.ge(iou,0)]

            
        losses = -torch.log(iou + self._eps)
#         print('iou2', iou)
#         print('iou2_loss', losses)
#         print('area_intersect', area_intersect)
#         print('area_union', area_union)
#        print('iou', iou)
#         print('weight', weight)
#         print('before weight losses', losses)

        if weight is not None:            
            losses = losses * weight    
#         print('after weight losses', losses)
        assert losses.numel() != 0
        return losses.mean()