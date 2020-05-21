"""
    By.Wheat
    2020.05.14
"""
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *
from layers.box_utils import decode, nms, match
import os

from base_nets import *


class SSDFace(SSDBackbone):
    """
    Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    vgg300_cfg = {
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'multi_box_nums': [4, 6, 6, 6, 4, 4],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [30, 60, 111, 162, 213, 264],
        'max_sizes': [60, 111, 162, 213, 264, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'SSD300_VGG',
    }

    def __init__(self, base, cfg=None, is_cuda=torch.cuda.is_available()):
        super(SSDFace, self).__init__(base, is_cuda)
        if cfg is None:
            cfg = self.vgg300_cfg
        # configure information
        self.cfg = cfg
        self.size = cfg['min_dim']
        self.variance = cfg['variance']
        self.priors = Variable(PriorBox(cfg).forward(), volatile=True)
        self.is_cuda = is_cuda

        # --Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)

        # --head layers
        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()
        self.convolution_predictor()

        self.name = type(self).__name__ + str(self.size) + '_' + type(base).__name__
        print(self.name, "init finished")

    def convolution_predictor(self):
        assert len(self.cfg['feature_maps']) == len(self.detect_blocks) and \
               len(self.cfg['multi_box_nums']) == len(self.detect_blocks)
        for (block, box_num) in zip(self.detect_blocks, self.cfg['multi_box_nums']):
            in_channels = block[1]
            loc, _ = repeat_conv2d(1, in_channels, box_num*4, 3, padding=1, activation=None)
            conf, _ = repeat_conv2d(1, in_channels, box_num, 3, padding=1, activation=None)
            self.loc += loc
            self.conf += conf

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:

            list of concat outputs from:
                1: localization layers, Shape: [batch_num,priors_num,4]
                2: confidence layers, Shape: [batch_num,priors_num]
                3: priorboxes, Shape: [priors_num,4]
        """
        sources = self.forward_for_detect_source(x)
        sources[0] = self.L2Norm(sources[0])
        loc = []
        conf = []

        # apply convolution_predictor to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat(tuple(o.view(o.size(0), -1) for o in loc), 1)
        conf = torch.cat(tuple(o.view(o.size(0), -1) for o in conf), 1)

        return (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1),
                self.priors.type_as(x))

    def detect(self, x, top_k=200, conf_thresh=0.4, nms_thresh=0.45):

        (loc_data, conf_data, priors) = self.forward(x)
        conf_data = torch.sigmoid(conf_data)
        """
            1: loc_data, Shape: [batch_num,priors_num,4]
            2: conf_data, Shape: [batch_num,priors_num]
            3: priors_data, Shape: [priors_num,4]
        """
        batch_num = loc_data.size(0)
        priors_num = priors.size(0)
        if top_k is None or top_k <= 0:
            top_k = priors_num
        output = []

        # Decode predictions into bboxes.
        for i in range(batch_num):
            decoded_boxes = decode(loc_data[i], priors, self.variance)
            # For each class, perform nms
            conf_scores = conf_data[i].clone()

            c_mask = conf_scores.gt(conf_thresh)
            scores = conf_scores[c_mask]
            if scores.size(0) == 0:
                output += [torch.Tensor()]
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            # idx of highest scoring and non-overlapping boxes per class
            ids, count = nms(boxes, scores, nms_thresh, top_k)
            output += [torch.cat((scores[ids[:count]].unsqueeze(1),
                                 boxes[ids[:count]]), 1)]

        return output

    def auto_load_weights(self, file):
        self.load_weights(self, file)


class SSDFaceLoss(nn.Module):
    def __init__(self, match_overlap, neg_pos_rate, do_neg_mining=True, variance=None, is_cuda=None):
        super(SSDFaceLoss, self).__init__()
        self.match_overlap = match_overlap
        self.neg_pos_rate = neg_pos_rate
        self.do_neg_mining = do_neg_mining
        if is_cuda is None:
            self.is_cuda = torch.cuda.is_available()
        else:
            self.is_cuda = is_cuda
        # variance is a parameter in encode/decode step, which should be equal to variance in SSDFace
        if variance is None:
            variance = [0.1, 0.2]
        self.variance = variance

    def forward(self, predictions, targets, conf_gain=1, thresh=0.4):
        """
        :param predictions: (loc_p, conf_p, priors)
                            loc_p, Shape: tensor[batch_num,priors_num,4]
                            conf_p, Shape: tensor[batch_num,priors_num]
                            priors, Shape: tensor[batch_num,4]
        :param targets: list[tensor[boxes_num, 5] for batch_num]
        :param conf_gain: Loss = Loss_loc + conf_gain * Loss_conf
        :return:
        """
        loc_p, conf_p, priors = predictions

        batch_num = loc_p.shape[0]
        priors_num = loc_p.shape[1]
        assert priors_num == priors.shape[0]

        loc_t = torch.Tensor(batch_num, priors_num, 4)
        matched = torch.LongTensor(batch_num, priors_num)
        for idx in range(batch_num):
            if len(targets[idx]) == 0:
                matched[idx] = 0
            else:
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                match(self.match_overlap, truths, priors, self.variance, labels,
                      loc_t, matched, idx)
        if self.is_cuda:
            loc_t = loc_t.cuda()
            matched = matched.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        matched = Variable(matched, requires_grad=False)

        """
        positive priors mask and negative priors mask:
          --pos_mask, neg_mask: [batch_size, priors_num]
          --pos_loc_mask, neg_loc_mask: [batch_size, priors_num, 4]
        """
        pos_mask = matched > 0
        pos_loc_mask = pos_mask.unsqueeze(-1).expand(batch_num, priors_num, 4)

        """calculate Localization Loss"""
        pos_loc_t = loc_t[pos_loc_mask].view(-1, 4)
        pos_loc_p = loc_p[pos_loc_mask].view(-1, 4)
        loc_loss = F.smooth_l1_loss(pos_loc_p, pos_loc_t, reduction='sum')

        """calculate Confidence Loss"""
        conf_loss_matrix = conf_p.clone()
        conf_loss_matrix[pos_mask] = -torch.log(torch.sigmoid(conf_loss_matrix[pos_mask]))
        conf_loss_matrix[~pos_mask] = -torch.log(1 - torch.sigmoid(conf_loss_matrix[~pos_mask]))
        #   --the Confidence Loss of positive priors. Calculate directly.
        pos_conf_loss = conf_loss_matrix[pos_mask].sum()
        #   --the Confidence Loss of negative priors. Do Hard Negative Mining: delete some negative priors
        if self.do_neg_mining:
            # conf_loss_matrix[pos_mask] = 0
            neg_conf_loss_matrix = torch.zeros_like(conf_loss_matrix)
            neg_conf_loss_matrix[~pos_mask] = conf_loss_matrix[~pos_mask]
            conf_loss_matrix = neg_conf_loss_matrix
            _, neg_loss_idx = conf_loss_matrix.sort(1, descending=True)
            _, conf_loss_rank = neg_loss_idx.sort(1)
            max_neg_num_of_each_batch = pos_mask.sum(1, keepdim=True) * self.neg_pos_rate
            max_neg_num_of_each_batch.clamp_(max=priors_num)
            part_neg_mask = conf_loss_rank < max_neg_num_of_each_batch.expand_as(conf_loss_rank)
            neg_conf_loss = conf_loss_matrix[part_neg_mask].sum()
            pos_sum = pos_mask.sum().clamp_(min=1)
            conf_loss = (pos_conf_loss + neg_conf_loss) / pos_sum
        else:
            over_thresh_neg_mask = torch.sigmoid(conf_p).gt(thresh) & (~pos_mask)
            neg_conf_loss = conf_loss_matrix[over_thresh_neg_mask].sum()
            pos_sum = pos_mask.sum().clamp_(min=1)
            neg_sum = over_thresh_neg_mask.sum().clamp_(min=1)
            conf_loss = pos_conf_loss / pos_sum + neg_conf_loss * self.neg_pos_rate / neg_sum

        loc_loss /= pos_sum
        return loc_loss, conf_gain * conf_loss


if __name__ == '__main__':
    a = SSDFace(VGG(3))
    print(a.name)
