import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from FlashNet.facedet.utils.bbox.box_utils import match, match_distillation, log_sum_exp, decode

GPU = False
if torch.cuda.is_available():
    GPU = True


class AttentionDistillationLoss(nn.Module):
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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap,
                 encode_target):
        super(AttentionDistillationLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, predictions_tch, priors, targets):
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

        (loc_data, conf_data), feat_stu = predictions
        (loc_data_tch, conf_data_tch), feat_tch = predictions_tch

        feat_tch = torch.cat([o.view(o.size(0), o.size(1), -1) for o in feat_tch], -1).unsqueeze(-1)
        feat_tch = torch.cat([feat_tch, feat_tch], -1)
        feat_tch = feat_tch.view(feat_tch.size(0), feat_tch.size(1), -1).permute(0, 2, 1)

        feat_stu = torch.cat([o.view(o.size(0), o.size(1), -1) for o in feat_stu], -1).unsqueeze(-1)
        feat_stu = torch.cat([feat_stu, feat_stu], -1)
        feat_stu = feat_stu.view(feat_stu.size(0), feat_stu.size(1), -1).permute(0, 2, 1)

        # print('feat_tch.size()', feat_tch.size()) #[batch, 43008, 1]
        # print('feat_stu.size()', feat_stu.size())


        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        #         pred_boxes = torch.Tensor(num, num_priors, 4)
        #         for i in range(0, num):
        #             pred_boxes[i] = decode(loc_data_tch.data[i], priors.data, self.variance)

        # teacher model's pred boxes
        pred_boxes = list()
        for i in range(0, num):
            # print(loc_data_tch.data[i].size())
            pred_boxes.append(decode(loc_data_tch.data[i], priors.data, self.variance))
        #         print(pred_boxes)
        #         print(targets)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            tch_pred = pred_boxes[idx].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            # match_distillation(self.threshold, truths, tch_pred, defaults, self.variance, labels, loc_t, conf_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
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

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        batch_conf_tch = conf_data_tch.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # conf_p = conf_data[(pos_idx).gt(0)].view(-1, self.num_classes)

        conf_p_tch = conf_data_tch[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes).detach()
        # conf_p_tch = conf_data_tch[(pos_idx).gt(0)].view(-1, self.num_classes).detach()



#--------------------------------------------------------------------------------------------------------
        pos_idx = pos.unsqueeze(2).expand_as(feat_tch)
        neg_idx = neg.unsqueeze(2).expand_as(feat_tch)
        feat_tch = feat_tch[pos_idx + neg_idx].view(-1, feat_tch.size(2))#[1, n_samples, dim]
        # feat_tch = feat_tch[pos_idx].view(-1, feat_tch.size(2))  # [1, n_samples, dim]
        # feat_tch = feat_tch.pow(2).mean(1)
        feat_tch = feat_tch/feat_tch.max()
        # feat_tch[feat_tch.le(0.2)] = 0

        # print(feat_tch)
        # print(feat_tch.size())

        pos_idx = pos.unsqueeze(2).expand_as(feat_stu)
        neg_idx = neg.unsqueeze(2).expand_as(feat_stu)
        feat_stu = feat_stu[pos_idx + neg_idx].view(-1, feat_stu.size(2))
        # feat_stu = feat_stu[pos_idx].view(-1, feat_stu.size(2))
        # feat_stu = feat_stu.pow(2).mean(1)
        feat_stu = feat_stu/feat_stu.max()
        # print('feat_tch.size()', feat_tch.size())
        # print('feat_stu.size()', feat_stu.size())
        loss_mimic = F.smooth_l1_loss(feat_stu, feat_tch, size_average=False, reduce=False)
        # print('loss_mimic.size()',loss_mimic.size())
        # loss_mimic = F.smooth_l1_loss(feat_stu[feat_tch.gt(0.2)], feat_tch[feat_tch.gt(0.2)], size_average=False, reduce=True)/feat_tch.gt(0.2).sum()
        # print(feat_tch.gt(0.2).sum())
        # print(loss_mimic)

        # print(loss_mimic.size())
# --------------------------------------------------------------------------------------------------------


        # print('conf_data_tch.requires_grad', conf_p_tch.requires_grad)
        if conf_p.size(0) != 0:
            log_conf_p = F.log_softmax(conf_p, -1)
            log_conf_p_tch = F.log_softmax(conf_p_tch, -1)
            conf_p_tch = F.softmax(conf_p_tch, -1)
            # print('conf_p_tch.size()', conf_p_tch.size())
            # print('conf_p_tch', conf_p_tch)
            # print('feat_tch', feat_tch)

            # print(conf_p_tch)
            Tq = - conf_p_tch * log_conf_p_tch

            KL = (conf_p_tch * log_conf_p_tch - conf_p_tch * log_conf_p)

            KL_sum_expand = KL.sum(1).unsqueeze(1).expand_as(KL)
            Tq_sum_expand = Tq.sum(1).unsqueeze(1).expand_as(Tq)
            # print('Tq', Tq_sum_expand)
            # print('KL', KL_sum_expand)
            gamma = 2
            beta = 2
            pt = (1 - (-KL_sum_expand - beta * Tq_sum_expand).exp())
            weight = pt.pow(gamma)
            # weight = Variable(weight, requires_grad=False)
            loss_c_distillation = (KL * weight).sum()
            # print('KL.size()', KL.size())
            # print('weight.size()', weight.size())
            #
            # print(weight[:,0].unsqueeze(1).size())
            # print(weight[:,0])
            loss_mimic = (loss_mimic * weight[:, 0].unsqueeze(1)).sum()

            # loss_c_distillation = (conf_p_tch*log_conf_p_tch).sum() - (conf_p_tch*log_conf_p).sum()
        else:
            raise NotImplementedError("distillation loss has no sample!")
            loss_c_distillation = loss_c

            # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_c_distillation /= N
        loss_mimic /= N
        return loss_l, loss_c, loss_c_distillation, loss_mimic