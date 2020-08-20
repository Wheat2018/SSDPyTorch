# TODO merge naive and weighted loss.
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


# class WingLoss(nn.Module):
#
#     def __init__(self, w=10, epsilon=2, weight=None):
#         super(WingLoss, self).__init__()
#         self.w = w
#         self.epsilon = epsilon
#         self.C = self.w - self.w * np.log(1 + self.w / self.epsilon)
#         self.weight = weight
#
#     def forward(self, predictions, targets):
#         """
#         :param predictions: 网络输出，预测坐标，一个形状为[batch, N, 2]的张量 或 [batch, N×2]的张量
#         :param targets: 目标，真实坐标，一个形状为[batch, N, 2]的张量 或 [batch, N×2]的张量
#         :return: wing loss，标量
#         """
#         x = predictions - targets
#         if self.weight is not None:
#             x = x * self.weight
#         t = torch.abs(x)
#
#         return torch.mean(torch.where(t < self.w,
#                            self.w * torch.log(1 + t / self.epsilon),
#                            t - self.C))


def wing_loss(predictions, targets, w=10, epsilon=2, weight=None, reduction='mean'):
    """
    :param predictions: 网络输出，预测坐标，一个形状为[batch, N, 2]的张量 或 [batch, N×2]的张量
    :param targets: 目标，真实坐标，一个形状为[batch, N, 2]的张量 或 [batch, N×2]的张量
    :return: wing loss，标量
    """
    C = w - w * np.log(1 + w / epsilon)
    x = predictions - targets
    if weight is not None:
        x = x * weight
    t = torch.abs(x)

    loss = torch.where(t < w, w * torch.log(1 + t / epsilon), t - C)

    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def iou_loss(pred_bboxes, target_bboxes, reduction='mean'):
 
    ious = bbox_overlaps(pred_bboxes, target_bboxes, is_aligned=True)
    ious = ious[ious>0]
#     print('pred_bboxes',pred_bboxes)
#     print('target_bboxes',target_bboxes)
    

    loss = -ious.log()
#     print('ious',ious)
#     print('loss',loss)
#     import pdb
#     pdb.set_trace()
    
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
    

def weighted_nll_loss(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, reduction='none')
    return torch.sum(raw * weight)[None] / avg_factor


def weighted_cross_entropy(pred, label, weight, avg_factor=None, reduce=True):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, reduction='none')
    if reduce:
        return torch.sum(raw * weight)[None] / avg_factor
    else:
        return raw * weight / avg_factor


def weighted_binary_cross_entropy(pred, label, weight, avg_factor=None):
    if avg_factor is None:
        avg_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        reduction='sum')[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       #                        weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='elementwise_mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    #     import pdb
    #     pdb.set_trace()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target))
    #     weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    # print('weight[target>0]', weight[target > 0])
    #     import pdb
    #     pdb.set_trace()

    loss = F.binary_cross_entropy_with_logits(pred, target,
                                              reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='elementwise_mean')[None]


def smooth_l1_loss(pred, target, beta=1.0, reduction='elementwise_mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction == 0:
        return loss
    elif reduction == 1:
        return loss.sum() / pred.numel()
    elif reduction == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.
    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
                bboxes2[:, 3] - bboxes2[:, 1])
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious