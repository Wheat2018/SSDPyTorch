from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from FlashNet.facedet.utils.bbox.box_utils import match_focal_loss, log_sum_exp
GPU = False
if torch.cuda.is_available():
    GPU = True
    
#from utils import one_hot_embedding
from torch.autograd import Variable

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]


class MultiBoxFocalLoss(nn.Module):    
    def __init__(self, num_classes=1, overlap_thresh=0.5):
        super(MultiBoxFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.variance = [0.1, 0.2]
        self.threshold = overlap_thresh

    def sigmoid_focal_loss(self, pred,
                           target,
                           weight,
                           gamma=2.0,
                           alpha=0.25,
                           reduction='mean'):
        pred_sigmoid = pred.sigmoid()

        target = target.type_as(pred).unsqueeze(2)
#         print('pred.size()',pred.size())
#         print('pred_sigmoid.size()',pred_sigmoid.size())
#         print('target.size()',target.size())
        
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
#         print(weight)
#         print(weight.float().type())
        weight = (alpha * target + (1 - alpha) * (1 - target)) * weight.float()
        weight = weight * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * weight
        reduction_enum = F._Reduction.get_enum(reduction)
#         print('reduction_enum',reduction_enum)
        # none: 0, mean:1, sum: 2
        if reduction_enum == 0:
            return loss
        elif reduction_enum == 1:
            return loss.mean()
        elif reduction_enum == 2:
            return loss.sum()


    def weighted_sigmoid_focal_loss(self, pred,
                                    target,
                                    weight,
                                    gamma=2.0,
                                    alpha=0.25,
                                    avg_factor=None,
                                    num_classes=80):
        if avg_factor is None:
            print('weight.sum()',weight.sum())
            print('weight>0.sum()',(weight>0).sum())
            print(num_classes)
            avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
            print('avg_factor',avg_factor)
        return self.sigmoid_focal_loss(
            pred, target, weight, gamma=gamma, alpha=alpha,
            reduction='sum')[None] / avg_factor

#     def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
    def forward(self, predictions, priors, targets):
        '''Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          conf_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          conf_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(conf_preds, conf_targets).
        '''
        
        loc_preds, conf_preds, _ = predictions
        priors = priors
        num = loc_preds.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_targets = torch.Tensor(num, num_priors, 4)
        conf_targets = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match_focal_loss(self.threshold, truths, defaults, self.variance, labels, loc_targets, conf_targets, idx)
        if GPU:
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()
        # wrap targets
        loc_targets = Variable(loc_targets, requires_grad=False)
        conf_targets = Variable(conf_targets, requires_grad=False)
        batch_size, num_boxes = conf_targets.size()
#         print('conf_targets',conf_targets)
#         print('conf_targets.size()',conf_targets.size())
#         print('batch_size',batch_size)
#         print('num_boxes',num_boxes)      
        
        pos = conf_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
#         print('num_pos',num_pos)

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = conf_targets > -1  # exclude ignored anchors
        
        num_pos_neg = pos_neg.data.long().sum()
#         print('num_pos_neg',num_pos_neg)
#         print('conf_preds',conf_preds.size())        
        mask = pos_neg.unsqueeze(2).expand_as(conf_preds)        
        cls_loss = self.weighted_sigmoid_focal_loss(conf_preds, conf_targets, mask,avg_factor=num_pos.float(), num_classes=self.num_classes)

        
        masked_conf_preds = conf_preds[mask].view(-1,self.num_classes)
#         print('conf_preds',conf_preds)
#         print('conf_preds',conf_preds.size())
#         print('mask',mask.size())        
#         print('masked_conf_preds',masked_conf_preds.size())
#         print('conf_targets',conf_targets.size())
#         cls_loss = self.focal_loss_alt(masked_conf_preds, conf_targets[pos_neg])

#         print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0].float()/num_pos), end=' | ')
#         print(loc_loss.type())
#         print(cls_loss.type())
#         print(num_pos.type())
#         loss = (loc_loss.float()+cls_loss.float()).float()/num_pos.float()
        return loc_loss/num_pos.float(), cls_loss