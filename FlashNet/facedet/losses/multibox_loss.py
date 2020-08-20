import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from FlashNet.facedet.utils.bbox.box_utils import match, match_landmark, match_refined_conf, log_sum_exp, decode
from .losses import wing_loss
GPU = False
if torch.cuda.is_available():
    GPU = True


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_landmark=False, out_pos_idx=False):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.use_landmark = use_landmark
        self.refine_conf = False
        self.out_pos_idx = out_pos_idx

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        if self.use_landmark:
            landmark_data, loc_data, conf_data = predictions
            # print('landmark_data', landmark_data.size())
        else:
            loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        # match priors (default boxes) and ground truth boxes
        if self.use_landmark:
            landmark_t = torch.Tensor(num, num_priors, 10)
            # landmark_weight = torch.ones(num, num_priors, 10)
        if self.refine_conf:
            refine_conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, 4].data
            defaults = priors.data

            if self.use_landmark:
                landmark_truths = targets[idx][:, 5:].data
                match_landmark(self.threshold, truths, landmark_truths, defaults, self.variance, labels, loc_t, conf_t, landmark_t, idx)
            else:
                if self.refine_conf:
                    refined_priors = decode(loc_data[idx].data, defaults.data, self.variance)
                    match_refined_conf(self.threshold, truths, defaults, refined_priors, self.variance, labels, loc_t, conf_t, refine_conf_t, idx)
                else:
                    match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            if self.use_landmark:
                landmark_t = landmark_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # print('conf_t.size()', conf_t.size())
        # print('landmark_t.size()',(landmark_t != -1).all(dim=-1).size())
        # print('conf_t > 0',conf_t > 0)
        # print(landmark_t)
        # print('(landmark_t != -1).all(dim=1)',(landmark_t != -1).all(dim=1).size())
        if self.use_landmark:
            # print('(conf_t > 0).size()', (conf_t > 0).size())
            # print('(landmark_t != -1).all(dim=-1)',(landmark_t != -1).all(dim=-1).size())
            landmark_pos = (conf_t > 0)*(landmark_t != -1).all(dim=-1)
            # print('landmark_pos.sum()',landmark_pos.sum())
            landmark_pos_idx = landmark_pos.unsqueeze(landmark_pos.dim()).expand_as(landmark_data)
            landmark_p = landmark_data[landmark_pos_idx].view(-1, 10)
            landmark_t = landmark_t[landmark_pos_idx].view(-1, 10)

            # print('landmark_t', landmark_t.requires_grad)

            # loss_landmark = F.smooth_l1_loss(landmark_p, landmark_t, size_average=False)
            # loss_landmark = F.l1_loss(landmark_p, landmark_t, size_average=False)
            loss_landmark = wing_loss(landmark_p, landmark_t, w=10, reduction='sum')
            # print('loss_landmark', loss_landmark)
            # print('loss_landmark_1',loss_landmark)
            # print('loss_landmark', loss_landmark.item(), 'loss_landmark_l1', loss_landmark_l1.item())
            # print('loss_landmark',loss_landmark)
            # print('box', pos.long().sum())
            # print('landmark', landmark_pos.long().sum())


        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
#         print('pos.size()',pos.size())
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#         print('pos_idx.size()',pos_idx)
#         print('conf_data.size()',conf_data.size())
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        if self.out_pos_idx:
            if self.use_landmark:
                loss_landmark /= max(landmark_pos.long().sum().float(), 1)
                return loss_landmark, loss_l, loss_c, pos
            else:
                return loss_l, loss_c, loc_t, pos
        else:
            if self.use_landmark:
                loss_landmark /= max(landmark_pos.long().sum().float(), 1)
                return loss_landmark, loss_l, loss_c
            else:
                return loss_l, loss_c