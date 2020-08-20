import torch
import numpy as np
import pdb

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    # valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    # best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
    # best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if best_prior_overlap.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    # for j in range(best_prior_idx.size(0)):
    #     best_truth_idx[best_prior_idx[j]] = j
    for j in range(best_prior_idx.size(0)):
        # ignore hard gt(iou < 0.2)
        # import pdb
        # pdb.set_trace()
        if best_prior_overlap[j] >= 0.2:
            best_truth_overlap[best_prior_idx[j]] = 2 # ensure best prior
            best_truth_idx[best_prior_idx[j]] = j


    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

#     print(best_truth_overlap < 0.45)
#     print((best_truth_overlap < 0.45) & (best_truth_overlap > 0.35))
#     conf[(best_truth_overlap < 0.45) & (best_truth_overlap > 0.35)] = -1  # label as ignored
        
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def match_landmark(threshold, truths, landmark_truths, priors, variances, labels, loc_t, conf_t, landmark_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landmark_t: (tensor) Tensor to be filled w/ endcoded landmark targets.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    # valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    # best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
    # best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if best_prior_overlap.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)


    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        # ignore hard gt(iou < 0.2)
        # import pdb
        # pdb.set_trace()
        if best_prior_overlap[j] >= 0.2:
            best_truth_overlap[best_prior_idx[j]] = 2 # ensure best prior
            best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    # import pdb
    # pdb.set_trace()
    # print(matches[best_truth_overlap > threshold]*640)
    # print(point_form(priors)[best_truth_overlap > threshold]*640)
    conf = labels[best_truth_idx]  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    matches_landmark = landmark_truths[best_truth_idx]
    matches_landmark[best_truth_overlap < threshold] = -1.0

    # neg_landmark = (best_truth_overlap < threshold)
    # #     print(best_truth_overlap < 0.45)
    # #     print((best_truth_overlap < 0.45) & (best_truth_overlap > 0.35))
    # #     conf[(best_truth_overlap < 0.45) & (best_truth_overlap > 0.35)] = -1  # label as ignored
    #
    #
    # neg_mask = (matches_landmark == -1).all(dim=1)
    # pos_mask = (matches_landmark != -1).all(dim=1)
    # pos_mask = (best_truth_overlap > threshold)
    #
    # for i in range(0, 5):
    #     matches_landmark[:, 2 * i] = (matches_landmark[:, 2*i] - priors[:, 0]) / priors[:, 2]
    #     matches_landmark[:, 2 * i + 1] = (matches_landmark[:, 2 * i + 1] - priors[:, 1]) / priors[:, 3]
    #
    #
    # matches_landmark[neg_mask] = -1.0
    #
    # print('3', matches_landmark[pos_mask].max())
    # if matches_landmark[pos_mask].size(0)!=0 and matches_landmark[pos_mask].max()>2:
    #     print(matches_landmark[pos_mask].max())
    #     import pdb
    #     pdb.set_trace()
    #     print(point_form(priors)[pos_mask])
    #     print(matches[pos_mask])
    #
    #
    # landmark = matches_landmark

    # landmark = encode_landmark(matches_landmark, priors, matches)
    landmark = encode_ldmk(matches_landmark, priors)
    # landmark = encode_landmark_by_meanface(matches_landmark, priors, neg_landmark)
    loc = encode(matches, priors, variances)
    landmark_t[idx] = landmark # [num_priors,10] encoded landmark offsets to learn
    loc_t[idx] = loc  # [num_priors,4] encoded bbox offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior



def match_refined_conf(threshold, truths, priors, refined_priors, variances, labels, loc_t, conf_t, refined_conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landmark_t: (tensor) Tensor to be filled w/ endcoded landmark targets.
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    # valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    # best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
    # best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if best_prior_overlap.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    # best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        # ignore hard gt(iou < 0.2)
        if best_prior_overlap[j] >= 0.2:
            best_truth_overlap[best_prior_idx[j]] = 2 # ensure best prior
            best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx]  # Shape: [num_priors,4]

    # print(matches[best_truth_overlap > threshold]*640)
    # print(point_form(priors)[best_truth_overlap > threshold]*640)
    conf = labels[best_truth_idx]  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

    loc = encode(matches, priors, variances)

    loc_t[idx] = loc  # [num_priors,4] encoded bbox offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def match_distillation(threshold, truths, pred_boxes, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    
    overlaps_tch = jaccard(
        truths,
        pred_boxes
    )
    # (Bipartite Matching)
    # [num_objects, 1] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_overlap_tch, best_prior_idx_tch = overlaps_tch.max(1, keepdim=True)
#     print(point_form(priors)
#     import pdb
#     pdb.set_trace()
    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
    best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if best_prior_overlap.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_prior_idx_tch.squeeze_(1)
    best_prior_overlap_tch.squeeze_(1)
    # index_fill_(dim, index, val) 按照index，将val的值填充self的dim维度。
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    best_truth_overlap.index_fill_(0, best_prior_idx_tch, 2.5)  # ensure best prior

    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
        
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    
    tch_truths = pred_boxes[best_prior_idx_tch]
    # ignore bad results(iou < 0.5 with gt)  predict by teacher
    tch_truths[best_prior_overlap_tch < 0.5] = truths[best_prior_overlap_tch < 0.5]
    
#     matches_tch = pred_boxes[best_prior_idx_tch][best_truth_idx]
    matches_tch = tch_truths[best_truth_idx]
#     pdb.set_trace()
    matches = (matches + matches_tch)/2

    
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

#     print(best_truth_overlap < 0.45)
#     print((best_truth_overlap < 0.45) & (best_truth_overlap > 0.35))
#     conf[(best_truth_overlap < 0.45) & (best_truth_overlap > 0.35)] = -1  # label as ignored
        
    loc = encode(matches, priors, variances)
#     loc_distillation = encode(matches_tch, priors, variances)
    
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
#     loc_t_distillation[idx] = loc_distillation    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    
    
def match_focal_loss(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
    best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if best_prior_overlap.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
#     print('labels',labels)
    conf = labels[best_truth_idx]         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

#     print(best_truth_overlap < 0.45)
#     print((best_truth_overlap < 0.45) & (best_truth_overlap > 0.35))
    conf[(best_truth_overlap < 0.4) & (best_truth_overlap > 0.35)] = -1  # label as ignored
        
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landmark(matched, priors, matched_box):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4]. [xc, yc, w, h]
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    # mask landmarks with -1
    # import pdb
    # pdb.set_trace()
    neg_mask = (matched == -1).all(dim=1)
    pos_mask = (matched != -1).all(dim=1)
    # print('before matched[pos_mask]', matched[pos_mask]*640)
    # print('before matched_box[pos_mask]', matched_box[pos_mask]*640)
    # print('before priors[pos_mask]', point_form(priors)[pos_mask]*640)
    # print('match_overlaps', match_overlaps)
    # overlaps = jaccard(matched_box[pos_mask], point_form(priors)[pos_mask])
    # if overlaps.size(0)!=0:
    #     best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    #     print('best_truth_overlap', best_truth_overlap)

    for i in range(0, 5):
        # print('1',((matched[:, 2*i] - priors[:, 0])/ priors[:, 2]).max())
        # print('2',((matched[:, 2*i+1] - priors[:, 1])/ priors[:, 3]).max())
        matched[:, 2 * i] = (matched[:, 2*i] - priors[:, 0]) / priors[:, 2]
        matched[:, 2 * i + 1] = (matched[:, 2 * i + 1] - priors[:, 1]) / priors[:, 3]

        # if matched[:, 2 * i].min() < -100000 or matched[:, 2 * i+1].min() < -100000:
        #     import pdb
        #     pdb.set_trace()

    matched[neg_mask] = -1.0
    # print('matched[pos_mask]', matched[pos_mask])
    # import pdb
    # pdb.set_trace()
    # print('22', matched[pos_mask])
    # print('1', matched[pos_mask])
    # print('2', point_form(priors)[pos_mask]*640)
    # print('3', matched_box[pos_mask]*640)
    # if matched[pos_mask].size(0)!=0 and matched[pos_mask].max()>2:
    #     print(matched[pos_mask].max())
    #     import pdb
    #     pdb.set_trace()
    # print('pos_mask.sum()', pos_mask.sum())

    # return target for smooth_l1_loss
    return matched  # [num_priors, 10]


# def encode_landmark_by_meanface(matched, priors, neg_landmark):
#     """Encode the variances from the priorbox layers into the ground truth boxes
#     we have matched (based on jaccard overlap) with the prior boxes.
#     Args:
#         matched: (tensor) Coords of ground truth for each prior in point-form
#             Shape: [num_priors, 4].
#         priors: (tensor) Prior boxes in center-offset form
#             Shape: [num_priors,4].[xc, yc, w, h]
#         variances: (list[float]) Variances of priorboxes
#     Return:
#         encoded boxes (tensor), Shape: [num_priors, 4]
#     """
#
#     # dist b/t match center and prior's center
#     # mask landmarks with -1
#     # [xc, yc, w, h]->[xmin, ymin, xmax, ymax]
#     w = priors[:, 2]
#     h = priors[:, 3]
#     priors = point_form(priors)
#     mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
#     mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]
#
#     neg_mask = (matched == -1).all(dim=1)
#     pos_mask = (matched != -1).all(dim=1)
#
#     for i in range(0, 5):
#
#         matched[:, 2 * i] = matched[:, 2 * i] - (priors[:, 0] + mean_face_shape_x[i] * w) / w
#         matched[:, 2 * i + 1] = matched[:, 2 * i + 1] - (priors[:, 1] + mean_face_shape_y[i] * h) / h
#
#     matched[neg_mask] = -1.0
#     # print('matched[pos_mask]', matched[pos_mask])
#     # print('pos_mask.sum()', pos_mask.sum())
#     # import pdb
#     # pdb.set_trace()
#     # return target for smooth_l1_loss
#     return matched  # [num_priors, 10]

def decode_landmark(landmark_deltas, priors):
    """Decode landmark predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        landmark_deltas (tensor): location predictions for loc layers,
            Shape: [num_priors,10]   format: [x1,y1,...,x5,y5]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4]    format: [xc,yc,width,height]

    Return:
        decoded landmark predictions
    """
    # landmark_deltas = landmark_deltas.clamp(-1,1)

    for i in range(0, 5):
        landmark_deltas[:, 2*i] = landmark_deltas[:, 2*i] * priors[:, 2] + priors[:, 0]
        landmark_deltas[:, 2*i+1] = landmark_deltas[:, 2*i+1] * priors[:, 3] + priors[:, 1]

    return landmark_deltas



def encode_ldmk(ldmk, anchor):
    variances = [0.1, 0.2]
    # import pdb
    # pdb.set_trace()
    # pos_mask = (ldmk != -1).all(dim=1)
    neg_mask = (ldmk == -1).all(dim=1)
    index_x = torch.Tensor([0, 2, 4, 6, 8]).long()
    index_y = torch.Tensor([1, 3, 5, 7, 9]).long()
    # import pdb
    # pdb.set_trace()
    ldmk[:, index_x] = (ldmk[:, index_x] - anchor[:, 0].view(-1, 1)) / (anchor[:, 2].view(-1, 1) * variances[0])
    ldmk[:, index_y] = (ldmk[:, index_y] - anchor[:, 1].view(-1, 1)) / (anchor[:, 3].view(-1, 1) * variances[0])
    ldmk[neg_mask] = -1.0
    # print(ldmk[pos_mask])
    # pdb.set_trace()
    return ldmk


def decode_ldmk(ldmk, anchor):
    variances = [0.1, 0.2]
    index_x = torch.Tensor([0, 2, 4, 6, 8]).long()
    index_y = torch.Tensor([1, 3, 5, 7, 9]).long()
    ldmk[:, index_x] = ldmk[:, index_x] * variances[0] * anchor[:, 2].view(-1, 1) + anchor[:, 0].view(-1, 1)
    ldmk[:, index_y] = ldmk[:, index_y] * variances[0] * anchor[:, 3].view(-1, 1) + anchor[:, 1].view(-1, 1)
    return ldmk


def encode_ldmk_stage2(gt_ldmk, gt_boxes):
    variances = [0.1, 0.2]
    # pos_mask = (ldmk != -1).all(dim=1)
    neg_mask = (ldmk == -1).all(dim=1)
    index_x = torch.Tensor([0, 2, 4, 6, 8]).long()
    index_y = torch.Tensor([1, 3, 5, 7, 9]).long()
    # import pdb
    # pdb.set_trace()
    ldmk[:, index_x] = (ldmk[:, index_x] - anchor[:, 0].view(-1, 1)) / (anchor[:, 2].view(-1, 1) * variances[0])
    ldmk[:, index_y] = (ldmk[:, index_y] - anchor[:, 1].view(-1, 1)) / (anchor[:, 3].view(-1, 1) * variances[0])
    ldmk[neg_mask] = -1.0
    # print(ldmk[pos_mask])
    # pdb.set_trace()
    return ldmk

# def decode_landmark_by_meanface(landmark_deltas, priors):
#     """Decode landmark predictions using priors to undo
#     the encoding we did for offset regression at train time.
#     Args:
#         landmark_deltas (tensor): location predictions for loc layers,
#             Shape: [num_priors,10]   format: [x1,y1,...,x5,y5]
#         priors (tensor): Prior boxes in center-offset form.
#             Shape: [num_priors,4]    format: [xc,yc,width,height]
#
#     Return:
#         decoded landmark predictions
#     """
#     w = priors[:, 2]
#     h = priors[:, 3]
#     priors = point_form(priors)
#     mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
#     mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]
#     for i in range(0, 5):
#         landmark_deltas[:, 2*i] = landmark_deltas[:, 2*i] * w + (priors[:, 0] + mean_face_shape_x[i] * w)
#         landmark_deltas[:, 2*i+1] = landmark_deltas[:, 2*i+1] * h + (priors[:, 1] + mean_face_shape_y[i] * h)
#
#     return landmark_deltas


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # convert to [minx, miny, maxx, maxy]
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_tvm(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
#     boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
#     boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]
#     import pdb
#     pdb.set_trace()
    xmin = boxes[: , 0] - boxes[: , 2] / 2
    ymin = boxes[: , 1] - boxes[: , 3] / 2
    xmax = boxes[: , 0] + boxes[: , 2] / 2
    ymax = boxes[: , 1] + boxes[: , 3] / 2
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def encode_anchor_free(matched, prior_coords, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    of_l = prior_coords[:,0] - matched[:,0]
    of_r = matched[:,2] - prior_coords[:,0]
    of_t = prior_coords[:,1] - matched[:,1]
    of_b = matched[:,3] - prior_coords[:,3]

    # return target for iou_loss
    return torch.cat([of_l, of_r, of_t, of_b], 1)  # [num_priors,4]

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode_anchor_free(loc, prior_coords, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        prior_coords (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,2].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    #[minx, miny, maxx, maxy]
#     import pdb
#     pdb.set_trace()
    print('prior_coords[:, 0].size()',prior_coords[:, 0].size())
    print('torch.exp(loc[:, 0] * variances[1]).size()',torch.exp(loc[:, 0] * variances[1]).size())
    xmin = prior_coords[:, 0] - torch.exp(loc[:, 0] * variances[1])
    ymin = prior_coords[:, 1] - torch.exp(loc[:, 1] * variances[1])
    xmax = prior_coords[:, 0] + torch.exp(loc[:, 2] * variances[1])
    ymax = prior_coords[:, 1] + torch.exp(loc[:, 3] * variances[1])
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim = 1)
    return boxes

# Adapted from https://github.com/Hakuyume/chainer-ssd
def refine_anchor_decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
#     # convert to [minx, miny, maxx, maxy]
#     boxes[:, :2] -= boxes[:, 2:] / 2
#     boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
#     import pdb
#     pdb.set_trace()
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


def match_fcos(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_overlap = best_prior_overlap[valid_gt_idx, :]
    best_prior_idx = best_prior_idx[valid_gt_idx, :]
    if best_prior_overlap.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background

#     print(best_truth_overlap < 0.45)
#     print((best_truth_overlap < 0.45) & (best_truth_overlap > 0.35))
#     conf[(best_truth_overlap < 0.45) & (best_truth_overlap > 0.35)] = -1  # label as ignored
        
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior