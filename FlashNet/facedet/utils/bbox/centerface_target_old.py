# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys
import cv2
import torch
import numpy as np
import logging
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

class CenterFaceTargetGenerator(object):
    """Generate CenterFace targets"""
    # def __init__(self, stages=5, stride=[8, 8, 8, 8, 8],
    #              valid_range=[(0, 32), (32, 64), (64, 128), (128, 256), (256, 640)], **kwargs):
    def __init__(self, stages=2, stride=(8, 16),
                 valid_range=((32, 128), (128, 256)), **kwargs):
        # def __init__(self, stages=3, stride=[8, 16, 32],
        #              valid_range=[(10, 48), (48, 96), (96, 160)], **kwargs):
        super(CenterFaceTargetGenerator, self).__init__(**kwargs)
        self._stages = stages
        self._stride = stride
        self._valid_range = valid_range

    def generate_targets(self, img, boxes, ldmks=None):
        """
        Args:
            img : [H, W, 3]
            boxes : [N, 5]
            ldmks: [N, 10]
        Return:
            cls_targets: [所有步长的特征图点的个数之和, num_classes=1]
            reg_mask: [所有步长的特征图点的个数之和, 1]
            ctr_offset_targets: [所有步长的特征图点的个数之和, 2]
            box_wh_targets: [所有步长的特征图点的个数之和, 2]
            optional:
                ldmk_targets: [所有步长的特征图点的个数之和, 10]
        """
        logging.error("ldmks.data: "+str(ldmks.data))
        _, rh, rw = img.shape
        boxes = torch.cat([torch.zeros((1, 5)), boxes], dim=0)  # for gt assign confusion
        use_ldmks = False
        if ldmks is not None:
            use_ldmks = True
            ldmks = torch.cat([-1.0*torch.ones((1, 10)), ldmks], dim=0)  # for gt assign confusion

        cls_targets = []
        box_targets = []
        ctr_targets = []

        reg_mask = []
        if use_ldmks:
            ldmk_targets = []

        for i in range(self._stages):
            stride = self._stride[i]
            fw, fh = rw, rh
            while stride > 1:
                fw = int(np.ceil(fw / 2))
                fh = int(np.ceil(fh / 2))
                stride /= 2

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
                ldmks = ldmks[inds]
                stride_ldmks = stride_ldmks[inds]

            boxes_x0, boxes_y0, boxes_x1, boxes_y1 = stride_boxes[:, 0], stride_boxes[:, 1], stride_boxes[:, 2], stride_boxes[:, 3]

            # [H, W, N]
            of_l = sx.unsqueeze(2) - boxes_x0.unsqueeze(0).unsqueeze(0)
            of_t = sy.unsqueeze(2) - boxes_y0.unsqueeze(0).unsqueeze(0)
            of_r = -(sx.unsqueeze(2) - boxes_x1.unsqueeze(0).unsqueeze(0))
            of_b = -(sy.unsqueeze(2) - boxes_y1.unsqueeze(0).unsqueeze(0))

            of_w = of_l + of_r
            of_h = of_t + of_b

            boxes_xc = 0.5 * (boxes_x0 + boxes_x1)
            boxes_yc = 0.5 * (boxes_y0 + boxes_y1)
            boxes_w = boxes_x1 - boxes_x0
            boxes_h = boxes_y1 - boxes_y0

            if use_ldmks:
                # ldmks_x0, ldmks_y0, ldmks_x1, ldmks_y1, ldmks_x2, ldmks_y2, ldmks_x3, ldmks_y3, ldmks_x4, ldmks_y4 = ldmks.T[ :, :]
                # neg_mask = (ldmks == -1).all(dim=1)
                pos_mask = (ldmks != -1).all(dim=1)
                # ldmks_offset = np.zeros_like(stride_ldmks)
                # ldmks_offset = torch.zeros_like(ldmks[pos_mask])
                # ldmk_targets = np.ones((fh, fw, 10)) * -1.0
                ldmk_target = torch.ones(fh, fw, 10) * -1.0
                stride_ldmks = stride_ldmks[pos_mask]
                tmp_boxes_xc = boxes_xc[pos_mask]
                tmp_boxes_yc = boxes_yc[pos_mask]
                tmp_boxes_w = boxes_w[pos_mask]
                tmp_boxes_h = boxes_h[pos_mask]

                # ldmks_offset[neg_mask] = -1.0

                #ldmks: [N, 10]
                y = tmp_boxes_yc.floor().long()
                x = tmp_boxes_xc.floor().long()
                for idx in range(0, 5):
                    # import pdb
                    # pdb.set_trace()
                    ldmk_target[y, x, 2 * idx] = (stride_ldmks[:, 2 * idx] - tmp_boxes_xc) / tmp_boxes_w
                    ldmk_target[y, x, 2 * idx + 1] = (stride_ldmks[:, 2 * idx + 1] - tmp_boxes_yc) / tmp_boxes_h
                    # y = stride_ldmks[:, 2 * idx + 1].floor().long()
                    # x = stride_ldmks[:, 2 * idx].floor().long()
                    # if y.max() >=80 or x.max()>=80:
                    # import pdb
                    # pdb.set_trace()
                    # logging.info("y.data: " + str(y.data))
                    # logging.info("x.data: " + str(x.data))
                    # logging.info("ldmk_target[y, x, 2 * idx].data: "+str(ldmk_target[y, x, 2 * idx].data))
                    # logging.info("ldmks_offset[:, 2 * idx].data: "+str(ldmks_offset[:, 2 * idx].data))

                    # logging.error("y.data: " + str(y.data))
                    # logging.error("x.data: " + str(x.data))
                    # logging.error("ldmk_target[y, x, 2 * idx].data: "+str(ldmk_target[y, x, 2 * idx].data))
                    # logging.error("ldmks_offset[:, 2 * idx].data: "+str(ldmks_offset[:, 2 * idx].data))

                    # ldmk_target[y, x, 2 * idx] = ldmks_offset[:, 2 * idx]
                    # ldmk_target[y, x, 2 * idx + 1] = ldmks_offset[:, 2 * idx + 1]
                ldmk_targets.append(ldmk_target.view(-1, 10))
                # import pdb
                # pdb.set_trace()

                # for i, ldmk in enumerate(ldmks):
                #     ldmks_x0, ldmks_y0, ldmks_x1, ldmks_y1, ldmks_x2, ldmks_y2, ldmks_x3, ldmks_y3, ldmks_x4, ldmks_y4 = ldmk
                #     for j
                #     ldmk_targets[ldmks_y0, ldmks_x0, 0] = ldmks_offset[i][0]

            # 下采样误差
            offset_x = boxes_xc - boxes_xc.floor() - 0.5
            offset_y = boxes_yc - boxes_yc.floor() - 0.5

            # 当前位置与其中一个gt中心的距离
            of_x = sx.unsqueeze(2) - boxes_xc.floor().unsqueeze(0).unsqueeze(0)
            of_y = sy.unsqueeze(2) - boxes_yc.floor().unsqueeze(0).unsqueeze(0)


            sigma_x = ((of_w - 1) * 0.5 - 1) * 0.3 + 0.8
            s_x = 2 * (sigma_x ** 2)
            sigma_y = ((of_h - 1) * 0.5 - 1) * 0.3 + 0.8
            s_y = 2 * (sigma_y ** 2)
            # [H, W, N]
            guassian = torch.exp(-(of_x) ** 2 / s_x - (of_y) ** 2 / s_y)
            # 排除第一个gt:w=h=0
            cls_target = guassian[:, :, 1:].max(dim=-1)[0]
            cls_targets.append(cls_target.view(-1, 1))

            # im_cls_target = guassian[:,:,1:].max(dim=-1)[0].numpy()
            # im_cls_target = im_cls_target * 255.0
            # boxes_numpy=stride_boxes.numpy()[1:, :]
            # for (xmin, ymin, xmax, ymax) in boxes_numpy:
            #     cv2.rectangle(im_cls_target, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 255), 1)
            # cv2.imwrite("cls_gt.png", im_cls_target)
            # import pdb
            # pdb.set_trace()


            # im_cls_target[im_cls_target > 0] = 255.0
            # cv2.imwrite("cls_gt_1.png", im_cls_target)
            # show_image_pixel(im_cls_target)
            # import pdb
            # pdb.set_trace()

            syx = torch.stack((sy.view(-1), sx.view(-1))).transpose(1, 0).long()
            min_vr, max_vr = self._valid_range[i]
            #             [FH*FW, N]
            # 4个offset都大于零才表示这个点在gt box里
            # [FH*FW, N, 4]
            # [H, W, N, 4]
            offsets = torch.cat([of_l.unsqueeze(-1), of_t.unsqueeze(-1),
                                 of_r.unsqueeze(-1), of_b.unsqueeze(-1)], dim=-1)
            # import pdb
            # pdb.set_trace()
            of_byx = offsets[syx[:, 0], syx[:, 1]]
            is_in_box = (torch.prod(of_byx > 0, dim=2) == 1)
            box_scale = torch.sqrt((of_byx[:, :, 0] + of_byx[:, :, 2]) * (of_byx[:, :, 1] + of_byx[:, :, 3]))
            #            print('box_scale',box_scale)
            # is_valid_area = (box_scale > min_vr) * (box_scale < max_vr)
            # [FH*FW, N]
            # valid_pos = is_in_box * is_valid_area
            valid_pos = is_in_box
            of_valid = torch.zeros((fh, fw, num_gt))
            of_valid[syx[:, 0], syx[:, 1], :] = valid_pos.float()  # 1, 0
            of_valid[:, :, 0] = 0
            gt_inds = torch.argmax(of_valid, dim=-1)
            gt_inds[torch.argmax(of_valid, dim=-1) == torch.argmin(of_valid, dim=-1)] = 0
            # [FH*FW, N] -> [FH*FW, 1]
            offset_w, offset_h = (of_w * of_valid).max(-1)[0], (of_h * of_valid).max(-1)[0]
            # [FH, FW] -> [FH, FW, 2]
            offset_wh = torch.cat([offset_w.unsqueeze(-1), offset_h.unsqueeze(-1)], dim=-1)
            box_targets.append(offset_wh.view(-1, 2))
            # import pdb
            # pdb.set_trace()
            offset_xy = torch.cat([offset_x[gt_inds[syx[:, 0], syx[:, 1]]].unsqueeze(-1),
                                   offset_y[gt_inds[syx[:, 0], syx[:, 1]]].unsqueeze(-1)], dim=-1)


            # gt_inds_np = gt_inds.numpy()
            # gt_inds_np[gt_inds_np>0]=255
            # cv2.imwrite("gt_inds.jpg", gt_inds_np)
            #
            # offset_w[gt_inds>0]
            # offset_h[gt_inds > 0]
            # import pdb
            # pdb.set_trace()

            ctr_targets.append(offset_xy)
            reg_mask.append((gt_inds>0).view(-1))


        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        ctr_targets = torch.cat(ctr_targets, dim=0)
        reg_mask = torch.cat(reg_mask, dim=0)
        if use_ldmks:
            ldmk_targets = torch.cat(ldmk_targets, dim=0)
            # import pdb
            # pdb.set_trace()
            return cls_targets, box_targets, ctr_targets, reg_mask, ldmk_targets
        else:
            return cls_targets, box_targets, ctr_targets, reg_mask


        # print(cls_targets.size(), box_targets.size(), ctr_targets.size(), reg_mask.size())
        #torch.Size([10641, 1]) torch.Size([10641, 2]) torch.Size([10641, 2]) torch.Size([10641])

        # import pdb
        # pdb.set_trace()



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


    target_generator = CenterFaceTargetGenerator()
    cls_targets, box_targets, ctr_targets, reg_mask = target_generator.generate_targets(img, boxes, ldmks)

    print(cls_targets.max())
