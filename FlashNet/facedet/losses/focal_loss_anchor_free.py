from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from FlashNet.facedet.utils.bbox.box_utils import match_focal_loss, log_sum_exp

GPU = False
if torch.cuda.is_available():
    GPU = True

# from utils import one_hot_embedding
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
    return y[labels]  # [N,D]


class SingleFocalLoss(nn.Module):
    def __init__(self, num_classes=1, overlap_thresh=0.5):
        super(SingleFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.variance = [0.1, 0.2]
        self.threshold = overlap_thresh

    #     def sigmoid_focal_loss(self, pred,
    #                            target,
    #                            weight,
    #                            gamma=2.0,
    #                            alpha=0.75,
    #                            reduction='mean'):
    #         if pred.size(-1) == 1:
    #             pred_sigmoid = pred.sigmoid()
    #         else:
    #             pred_softmax = pred.softmax(dim=-1)[:,:,0:1]
    #
    # #         target = target.type_as(pred).unsqueeze(2)
    # #         print('pred.size()',pred.size())
    # #         print('pred_sigmoid.size()',pred_sigmoid.size())
    # #         print('target.size()',target.size())
    #
    #         pt = (1 - pred_softmax) * target + pred_softmax * (1 - target)
    # #         print(weight)
    # #         print(weight.float().type())
    # #        weight = (alpha * target + (1 - alpha) * (1 - target)) * weight.float()
    #         weight = (alpha * target + (1 - alpha) * (1 - target))
    #         weight = weight * pt.pow(gamma)
    #
    # #        loss = F.binary_cross_entropy_with_logits(
    # #            pred, target, reduction='none') * weight
    #         loss = F.cross_entropy(pred.view(-1,2), target.view(-1).long(), reduction='none')
    #         loss = loss*weight.view(-1)
    # #        print('weight',weight.size())
    # #        print('loss',loss.size())
    # #        import pdb
    # #        pdb.set_trace()
    #         reduction_enum = F._Reduction.get_enum(reduction)
    # #         print('reduction_enum',reduction_enum)
    #         # none: 0, mean:1, sum: 2
    #         if reduction_enum == 0:
    #             return loss
    #         elif reduction_enum == 1:
    #             return loss.mean()
    #         elif reduction_enum == 2:
    #             return loss.sum()

    def sigmoid_focal_loss(self,
                           pred,
                           target,
                           weight,
                           gamma=2.0,
                           alpha=0.25,
                           reduction='mean'):
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        import pdb
        pdb.set_trace()
        # weight = (alpha * target + (1 - alpha) * (1 - target)) * weight.float()
        weight = (alpha * target + (1 - alpha) * (1 - target))
        weight = weight * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction=reduction) * weight
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
            print('weight.sum()', weight.sum())
            print('weight>0.sum()', (weight > 0).sum())
            print(num_classes)
            avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
            print('avg_factor', avg_factor)

        print('num_pos:', avg_factor)
        return self.sigmoid_focal_loss(
            pred, target, weight, gamma=gamma, alpha=alpha,
            reduction='sum')[None] / avg_factor

    #     def forward(self, loc_preds, loc_targets, conf_preds, conf_targets):
    def forward(self, conf_preds, conf_targets):
        '''Compute loss between (conf_preds, conf_targets).
        Args:
          conf_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          conf_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = FocalLoss(conf_preds, conf_targets).
        '''
        # wrap targets
        conf_targets = Variable(conf_targets, requires_grad=False)
        #         batch_size, num_boxes = conf_targets.size()
        pos = conf_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        #         print('num_pos',num_pos)
        pos_neg = conf_targets > 0  # exclude ignored anchors
        num_pos_neg = pos_neg.data.long().sum()
        #         print('conf_preds.size()',conf_preds.size())
        #         print('pos_neg.unsqueeze(2)',pos_neg.size())
        mask = pos_neg.expand_as(conf_preds)
        cls_loss = self.weighted_sigmoid_focal_loss(conf_preds, conf_targets, mask, avg_factor=num_pos.float(),
                                                    num_classes=self.num_classes)

        return cls_loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha, num_class=1):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.num_class = num_class

    def forward(self, out, target):
        out = out.reshape(-1, self.num_class)
        target = target.reshape(-1)
        num_pos = torch.nonzero(target > 0).numel()
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
                -(t == class_ids).float() * alpha * term1
                - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum() / num_pos