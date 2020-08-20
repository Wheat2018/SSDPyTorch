# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys
import cv2
import torch
import numpy as np
import logging
import pdb
logging.basicConfig(filename='./log/centernet_target.log', level=logging.DEBUG)

def show_image_pixel(img):
    '''
    :param img: 需要输出像素值的图像，要求是灰度图
    :return: 无返回值
    '''
    height,width=img.shape
    for i in range(height):
        for j in range(width):
            print(' %3d' %img[i,j],end='')
        print('\n')

class KPFaceTargetGenerator(object):
    """Generate CenterFace targets"""
    # def __init__(self, stages=5, stride=[8, 8, 8, 8, 8],
    #              valid_range=[(0, 32), (32, 64), (64, 128), (128, 256), (256, 640)], **kwargs):
    def __init__(self, stages=2, stride=(8, 16),
                 valid_range=((32, 128), (128, 256)), **kwargs):
        # def __init__(self, stages=3, stride=[8, 16, 32],
        #              valid_range=[(10, 48), (48, 96), (96, 160)], **kwargs):
        super(KPFaceTargetGenerator, self).__init__(**kwargs)
        self._stages = stages
        self._stride = stride
        self._valid_range = valid_range

    def generate_targets(self, img, boxes, ldmks=None):
        """
        Args:
            img : [H, W, 3]
            boxes : [N, 5]
            ldmks: [N, 15]
        Return:
            cls_targets: [所有步长的特征图点的个数之和, num_classes=1]
            reg_mask: [所有步长的特征图点的个数之和, 1]
            ctr_offset_targets: [所有步长的特征图点的个数之和, 2]
            box_wh_targets: [所有步长的特征图点的个数之和, 2]
            optional:
                ldmk_targets: [所有步长的特征图点的个数之和, 10]
        """
        _, rh, rw = img.shape
        boxes = torch.cat([torch.zeros((1, 5)), boxes], dim=0)  # for gt assign confusion
        use_ldmks = False
        if ldmks is not None:
            use_ldmks = True
            # import pdb
            # pdb.set_trace()
            # print(ldmks)
            ldmks = torch.cat([-1.0*torch.ones((1, 15)), ldmks], dim=0)  # for gt assign confusion

        hard_face_targets = []
        box_targets = []
        box_mask = []
        ldmk_targets = []

        for i in range(self._stages):
            stride = self._stride[i]
            fw, fh = rw, rh
            while stride > 1:
                fw = int(np.ceil(fw / 2))
                fh = int(np.ceil(fh / 2))
                stride /= 2

            box_target = torch.Tensor(fh, fw, 4)

            rx = torch.arange(0, fw).view(1, -1)
            ry = torch.arange(0, fh).view(-1, 1)
            sx = rx.repeat(fh, 1).float()
            sy = ry.repeat(1, fw).float()
            # print('boxes', boxes)
            stride_boxes = (boxes[:, :4] / self._stride[i])
            # print('stride boxes', stride_boxes)
            num_gt = stride_boxes.size(0)
            areas = (stride_boxes[:, 2] - stride_boxes[:, 0]) * (stride_boxes[:, 3] - stride_boxes[:, 1])
            areas, inds = torch.sort(areas)
            stride_boxes = stride_boxes[inds]
            if use_ldmks:
                stride_ldmks = ldmks / self._stride[i]
                stride_ldmks[ldmks == -1] = -1
                stride_ldmks[:, 2::3] = ldmks[:, 2::3] # visible
                stride_ldmks = stride_ldmks[inds]

            boxes_x0, boxes_y0, boxes_x1, boxes_y1 = stride_boxes[:, 0], stride_boxes[:, 1], stride_boxes[:, 2], stride_boxes[:, 3]

            boxes_xc = 0.5 * (boxes_x0 + boxes_x1)
            boxes_yc = 0.5 * (boxes_y0 + boxes_y1)
            boxes_w = boxes_x1 - boxes_x0
            boxes_h = boxes_y1 - boxes_y0

            # [H, W, N]
            of_l = sx.unsqueeze(2) - boxes_x0.unsqueeze(0).unsqueeze(0)
            of_t = sy.unsqueeze(2) - boxes_y0.unsqueeze(0).unsqueeze(0)
            of_r = -(sx.unsqueeze(2) - boxes_x1.unsqueeze(0).unsqueeze(0))
            of_b = -(sy.unsqueeze(2) - boxes_y1.unsqueeze(0).unsqueeze(0))
            of_w = of_l + of_r
            of_h = of_t + of_b

            # 当前位置与其中一个gt中心的距离
            of_xc = sx.unsqueeze(2) - boxes_xc.floor().unsqueeze(0).unsqueeze(0)
            of_yc = sy.unsqueeze(2) - boxes_yc.floor().unsqueeze(0).unsqueeze(0)

            syx = torch.stack((sy.view(-1), sx.view(-1))).transpose(1, 0).long()
            min_vr, max_vr = self._valid_range[i]
            #             [FH*FW, N]
            # 4个offset都大于零才表示这个点在gt box里
            # [FH*FW, N, 4]
            # [H, W, N, 4]
            offsets = torch.cat([of_l.unsqueeze(-1), of_t.unsqueeze(-1),
                                 of_r.unsqueeze(-1), of_b.unsqueeze(-1)], dim=-1)
            of_byx = offsets[syx[:, 0], syx[:, 1]]
            is_in_box = (torch.prod(of_byx > 0, dim=2) == 1)
            box_scale = torch.sqrt((of_byx[:, :, 0] + of_byx[:, :, 2]) * (of_byx[:, :, 1] + of_byx[:, :, 3])) * self._stride[i]
            is_valid_area = (box_scale > min_vr) * (box_scale < max_vr)
            # [FH*FW, N]
            valid_pos = is_in_box * is_valid_area
            # valid_pos = is_in_box
            of_valid = torch.zeros((fh, fw, num_gt))
            of_valid[syx[:, 0], syx[:, 1], :] = valid_pos.float()  # 1, 0
            of_valid[:, :, 0] = 0
            gt_inds = torch.argmax(of_valid, dim=-1)
            gt_inds[torch.argmax(of_valid, dim=-1) == torch.argmin(of_valid, dim=-1)] = 0

            of_l = (of_l * of_valid).max(-1)[0]
            of_t = (of_t * of_valid).max(-1)[0]
            of_r = (of_r * of_valid).max(-1)[0]
            of_b = (of_b * of_valid).max(-1)[0]

            offsets = torch.cat([of_l.unsqueeze(-1),
                                 of_t.unsqueeze(-1),
                                 of_r.unsqueeze(-1),
                                 of_b.unsqueeze(-1)], dim=-1)

            box_targets.append(offsets.view(-1, 4))
            box_mask.append((gt_inds > 0).view(-1))
            # import pdb
            # pdb.set_trace()
            # box_targets[0][box_mask[0]]

            ldmks_with_valid_area = (torch.sqrt(areas)*self._stride[i] > min_vr)*(torch.sqrt(areas)*self._stride[i] < max_vr)
            ldmks_with_valid_anno = (ldmks[inds] != -1).all(dim=1)
            pos_ldmk_mask = ldmks_with_valid_area * ldmks_with_valid_anno
            # import pdb
            # pdb.set_trace()
            stride_ldmks = stride_ldmks[pos_ldmk_mask].view(-1, 5, 3).numpy()
            # print(self._stride[i], stride_ldmks)
            ldmk_target = gen_keypoint_heatmap(keypoints=stride_ldmks, output_res=fh, num_parts=5, sigma_scale_for_invisible=2)
            ldmk_target = torch.from_numpy(ldmk_target).permute(1, 2, 0)
            ldmk_targets.append(ldmk_target.view(-1, 5))

            # 一张脸如果少于3个关键点可见 则被定义为hard face
            ldmks_wo_valid_anno = torch.sum(ldmks[:, 2::3], dim=1) < 3
            pos_hard_face_mask = ldmks_with_valid_area * ldmks_wo_valid_anno
            if pos_hard_face_mask.sum() != 0:
                of_w = of_w[:, :, pos_hard_face_mask]
                of_h = of_h[:, :, pos_hard_face_mask]
                of_xc = of_xc[:, :, pos_hard_face_mask]
                of_yc = of_yc[:, :, pos_hard_face_mask]
                sigma_x = ((of_w - 1) * 0.5 - 1) * 0.3 + 0.8
                s_x = 2 * (sigma_x ** 2)
                sigma_y = ((of_h - 1) * 0.5 - 1) * 0.3 + 0.8
                s_y = 2 * (sigma_y ** 2)
                # [H, W, N]
                guassian = torch.exp(-(of_xc) ** 2 / s_x - (of_yc) ** 2 / s_y)
                # import pdb
                # pdb.set_trace()
                # 排除第一个gt:w=h=0
                # hard_face_target = guassian[:, :, 1:].max(dim=-1)[0]
                hard_face_target = guassian.max(dim=-1)[0]
                hard_face_targets.append(hard_face_target.view(-1, 1))
            else:
                hard_face_targets.append(torch.zeros((fh*fw, 1)))

        box_targets = torch.cat(box_targets, dim=0)
        box_mask = torch.cat(box_mask, dim=0)
        ldmk_targets = torch.cat(ldmk_targets, dim=0)
        hard_face_targets = torch.cat(hard_face_targets, dim=0)
        return box_targets, box_mask, ldmk_targets, hard_face_targets


def gen_keypoint_heatmap(keypoints, output_res, num_parts=5, sigma_scale_for_invisible=2):
    def gauss(hms, sigma, pt, idx, output_res):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        # print(x0, y0, sigma)
        # pdb.set_trace()
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        x, y = int(pt[0]), int(pt[1])
        if x >= 0 and y >= 0 and x < output_res and y < output_res:
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], output_res)
            aa, bb = max(0, ul[1]), min(br[1], output_res)

            hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b, c:d])

    sigma = output_res / 64

    sigma_scale_for_invisible = sigma_scale_for_invisible
    hms = np.zeros(shape=(num_parts, output_res, output_res), dtype=np.float32)
    for p in keypoints:
        for idx, pt in enumerate(p):
            if pt[2] == 1:  # 如果这个点可见
                tmp_sigma = sigma
                gauss(hms, tmp_sigma, pt, idx, output_res)
            if pt[2] == 0:  # 如果这个点不可见
                tmp_sigma = sigma * sigma_scale_for_invisible  # 对于不可见，但标注点，让heatmap高斯分布的方差变大，使得该点计算loss所占权重变小，或者该点容忍区域变大
                gauss(hms, tmp_sigma, pt, idx, output_res)
    return hms


def gaussian_radius(det_size, min_overlap=0.7):
    width, height = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0])
    mu_y = int(center[1])
    h, w = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def creat_roiheatmap(centern_roi, det_size_map):
    c_x, c_y = centern_roi
    sigma_x = ((det_size_map[1] - 1) * 0.5 - 1) * 0.3 + 0.8
    s_x = 2 * (sigma_x ** 2)
    sigma_y = ((det_size_map[0] - 1) * 0.5 - 1) * 0.3 + 0.8
    s_y = 2 * (sigma_y ** 2)
    X1 = np.arange(det_size_map[1])
    Y1 = np.arange(det_size_map[0])
    [X, Y] = np.meshgrid(X1, Y1)
    heatmap = np.exp(-(X - c_x) ** 2 / s_x - (Y - c_y) ** 2 / s_y)
    return heatmap


if __name__ == '__main__':
    img = torch.zeros(3, 600, 899)  # hwc
    #[600, 899] -> [75, 113] -> [38, 57]
    # boxes = torch.FloatTensor([[1, 88, 821, 480, 60],
    #                            [1, 75, 600, 251, 60],
    #                            [1, 235, 899, 598, 60],
    #                            [336, 48, 395, 117, 29]])

    boxes = torch.FloatTensor([[300.132, 88.85, 421.34, 180.61, 1],
                               [200.21, 200.26, 281.54, 380.73, 1]])

    ldmks = torch.FloatTensor([[330.12, 110.56, 393.48, 108.35, 360.54, 139.82, 340.91, 168.22, 383.66, 165.47],
                               [330.12, 110.56, 393.48, 108.35, 360.54, 139.82, 340.91, 168.22, 383.66, 165.47]])


    target_generator = KPFaceTargetGenerator()
    box_targets, box_mask, ldmk_targets = target_generator.generate_targets(img, boxes, ldmks)

