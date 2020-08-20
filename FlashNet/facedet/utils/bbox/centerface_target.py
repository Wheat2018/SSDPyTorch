# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys
import cv2
import torch
import numpy as np
import logging
import pdb
# logging.basicConfig(filename='./log/centernet_target.log', level=logging.DEBUG)

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
                 valid_range=((32, 128), (128, 256)), ldmk_stride=(4,), **kwargs):
        # def __init__(self, stages=3, stride=[8, 16, 32],
        #              valid_range=[(10, 48), (48, 96), (96, 160)], **kwargs):
        super(CenterFaceTargetGenerator, self).__init__(**kwargs)
        self._stages = stages
        self._stride = stride
        self._valid_range = valid_range

    def generate_targets(self, img, boxes, ldmks=None, ldmk_reg_type=None):
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
        # logging.error("ldmks.data: "+str(ldmks.data))
        # import pdb
        # pdb.set_trace()
        _, rh, rw = img.shape
        boxes = torch.cat([torch.zeros((1, 5)), boxes], dim=0)  # for gt assign confusion
        use_ldmks = False
        if ldmks is not None:
            use_ldmks = True
            # import pdb
            # pdb.set_trace()
            # print(ldmks)
            ldmks = torch.cat([-1.0*torch.ones((1, 15)), ldmks], dim=0)  # for gt assign confusion

        cls_targets = []
        box_targets = []
        ctr_targets = []

        box_mask = []
        if use_ldmks:
            ldmk_targets = []
            ldmk_mask = []

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


            # 下采样误差
            offset_x = boxes_xc - boxes_xc.floor() - 0.5
            offset_y = boxes_yc - boxes_yc.floor() - 0.5

            # 当前位置与其中一个gt中心的距离
            of_xc = sx.unsqueeze(2) - boxes_xc.floor().unsqueeze(0).unsqueeze(0)
            of_yc = sy.unsqueeze(2) - boxes_yc.floor().unsqueeze(0).unsqueeze(0)


            sigma_x = ((of_w - 1) * 0.5 - 1) * 0.3 + 0.8
            s_x = 2 * (sigma_x ** 2)
            sigma_y = ((of_h - 1) * 0.5 - 1) * 0.3 + 0.8
            s_y = 2 * (sigma_y ** 2)
            # [H, W, N]
            guassian = torch.exp(-(of_xc) ** 2 / s_x - (of_yc) ** 2 / s_y)
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
            of_byx = offsets[syx[:, 0], syx[:, 1]]
            is_in_box = (torch.prod(of_byx > 0, dim=2) == 1)
            # box_scale = torch.sqrt((of_byx[:, :, 0] + of_byx[:, :, 2]) * (of_byx[:, :, 1] + of_byx[:, :, 3]))
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
            offset_xy = torch.cat([offset_x[gt_inds[syx[:, 0], syx[:, 1]]].unsqueeze(-1),
                                   offset_y[gt_inds[syx[:, 0], syx[:, 1]]].unsqueeze(-1)], dim=-1)
            ctr_targets.append(offset_xy)
            import pdb
            pdb.set_trace()
            box_mask.append((gt_inds>0).view(-1))

            if ldmk_reg_type=='heatmap':
                pos_mask = (ldmks[inds] != -1).all(dim=1)
                stride_ldmks = stride_ldmks[pos_mask].view(-1, 5, 3).numpy()
                # print(self._stride[i], stride_ldmks)

                ldmk_target = gen_keypoint_heatmap(keypoints=stride_ldmks, output_res=fh, num_parts=5, sigma_scale_for_invisible=2)
                # for i in range(0, 5):
                #     heatmap = cv2.applyColorMap(np.uint8(255 * ldmk_target[i]), cv2.COLORMAP_JET)
                #     cv2.imwrite('heatmap_' + str(i) + '.jpg', heatmap)
                #
                # heatmap_all = cv2.applyColorMap(np.uint8(255 * (ldmk_target[0] + ldmk_target[1] + ldmk_target[2] + ldmk_target[3] + ldmk_target[4])), cv2.COLORMAP_JET)
                # cv2.imwrite('heatmap_all.jpg', heatmap_all)
                # pdb.set_trace()
                # chw -> hwc
                ldmk_target = torch.from_numpy(ldmk_target).permute(1, 2, 0)
                ldmk_targets.append(ldmk_target.view(-1, 5))

            if ldmk_reg_type=='coord':
                # ldmks_x0, ldmks_y0, ldmks_x1, ldmks_y1, ldmks_x2, ldmks_y2, ldmks_x3, ldmks_y3, ldmks_x4, ldmks_y4 = ldmks.T[ :, :]
                # neg_mask = (ldmks == -1).all(dim=1)
                pos_mask = (ldmks[inds] != -1).all(dim=1)
                ldmk_target = torch.ones(fh, fw, 10) * -1.0
                ldmk_target_mask = torch.zeros(fh, fw, 10)

                stride_ldmks = stride_ldmks[pos_mask]
                tmp_boxes_xc = boxes_xc[pos_mask]
                tmp_boxes_yc = boxes_yc[pos_mask]
                tmp_boxes_w = boxes_w[pos_mask]
                tmp_boxes_h = boxes_h[pos_mask]
                # ldmks_offset[neg_mask] = -1.0
                #ldmks: [N, 10]
                y = tmp_boxes_yc.floor().long()
                x = tmp_boxes_xc.floor().long()

                # (stride_ldmks[:, 2 * idx] - tmp_boxes_xc.round()) / tmp_boxes_w

                for idx in range(0, 5):
                    # import pdb
                    # pdb.set_trace()
                    ldmk_target[y, x, 2 * idx] = (stride_ldmks[:, 3 * idx] - tmp_boxes_xc.round()) / tmp_boxes_w
                    ldmk_target[y, x, 2 * idx + 1] = (stride_ldmks[:, 3 * idx + 1] - tmp_boxes_yc.round()) / tmp_boxes_h
                    # visible -1:未标注 0:不可见 1:可见
                    ldmk_target_mask[y, x, 2 * idx] = stride_ldmks[:, 3 * idx + 2]
                    ldmk_target_mask[y, x, 2 * idx + 1] = stride_ldmks[:, 3 * idx + 2]

                    # if ldmk_target.max()>1:
                    #     import pdb
                    #     pdb.set_trace()
                    #     print(tmp_boxes_xc.round())
                # import pdb
                # pdb.set_trace()
                ldmk_targets.append(ldmk_target.view(-1, 10))
                ldmk_mask.append((ldmk_target_mask==1).view(-1, 10))
                # import pdb
                # pdb.set_trace()
                # ldmk_mask.append((ldmk_target.max(-1)[0] != -1).view(-1))









            # if use_ldmks:
            #     pos_mask = (ldmks[inds] != -1).all(dim=1)
            #     neg_mask = (ldmks[inds] == -1).all(dim=1)
            #     of_valid[:, :, neg_mask] = 0
            #     gt_inds = torch.argmax(of_valid, dim=-1)
            #     gt_inds[torch.argmax(of_valid, dim=-1) == torch.argmin(of_valid, dim=-1)] = 0
            #     ldmk_mask.append((gt_inds > 0).view(-1))
            #
            #     of_x0 = stride_ldmks[:, 0].unsqueeze(0).unsqueeze(0) - sx.unsqueeze(2)
            #     of_x1 = stride_ldmks[:, 2].unsqueeze(0).unsqueeze(0) - sx.unsqueeze(2)
            #     of_x2 = stride_ldmks[:, 4].unsqueeze(0).unsqueeze(0) - sx.unsqueeze(2)
            #     of_x3 = stride_ldmks[:, 6].unsqueeze(0).unsqueeze(0) - sx.unsqueeze(2)
            #     of_x4 = stride_ldmks[:, 8].unsqueeze(0).unsqueeze(0) - sx.unsqueeze(2)
            #     of_y0 = stride_ldmks[:, 1].unsqueeze(0).unsqueeze(0) - sy.unsqueeze(2)
            #     of_y1 = stride_ldmks[:, 3].unsqueeze(0).unsqueeze(0) - sy.unsqueeze(2)
            #     of_y2 = stride_ldmks[:, 5].unsqueeze(0).unsqueeze(0) - sy.unsqueeze(2)
            #     of_y3 = stride_ldmks[:, 7].unsqueeze(0).unsqueeze(0) - sy.unsqueeze(2)
            #     of_y4 = stride_ldmks[:, 9].unsqueeze(0).unsqueeze(0) - sy.unsqueeze(2)
            #
            #     of_x0[:, :, neg_mask] = 0
            #     of_x1[:, :, neg_mask] = 0
            #     of_x2[:, :, neg_mask] = 0
            #     of_x3[:, :, neg_mask] = 0
            #     of_x4[:, :, neg_mask] = 0
            #     of_y0[:, :, neg_mask] = 0
            #     of_y1[:, :, neg_mask] = 0
            #     of_y2[:, :, neg_mask] = 0
            #     of_y3[:, :, neg_mask] = 0
            #     of_y4[:, :, neg_mask] = 0
            #
            #     of_x0[:, :, pos_mask] = (of_x0[:, :, pos_mask]) / of_w[:, :, pos_mask]
            #     of_x1[:, :, pos_mask] = (of_x1[:, :, pos_mask]) / of_w[:, :, pos_mask]
            #     of_x2[:, :, pos_mask] = (of_x2[:, :, pos_mask]) / of_w[:, :, pos_mask]
            #     of_x3[:, :, pos_mask] = (of_x3[:, :, pos_mask]) / of_w[:, :, pos_mask]
            #     of_x4[:, :, pos_mask] = (of_x4[:, :, pos_mask]) / of_w[:, :, pos_mask]
            #     of_y0[:, :, pos_mask] = (of_y0[:, :, pos_mask]) / of_h[:, :, pos_mask]
            #     of_y1[:, :, pos_mask] = (of_y1[:, :, pos_mask]) / of_h[:, :, pos_mask]
            #     of_y2[:, :, pos_mask] = (of_y2[:, :, pos_mask]) / of_h[:, :, pos_mask]
            #     of_y3[:, :, pos_mask] = (of_y3[:, :, pos_mask]) / of_h[:, :, pos_mask]
            #     of_y4[:, :, pos_mask] = (of_y4[:, :, pos_mask]) / of_h[:, :, pos_mask]
            #
            #     offset_x0 = (of_x0 * of_valid).max(-1)[0]
            #     offset_x1 = (of_x1 * of_valid).max(-1)[0]
            #     offset_x2 = (of_x2 * of_valid).max(-1)[0]
            #     offset_x3 = (of_x3 * of_valid).max(-1)[0]
            #     offset_x4 = (of_x4 * of_valid).max(-1)[0]
            #     offset_y0 = (of_y0 * of_valid).max(-1)[0]
            #     offset_y1 = (of_y1 * of_valid).max(-1)[0]
            #     offset_y2 = (of_y2 * of_valid).max(-1)[0]
            #     offset_y3 = (of_y3 * of_valid).max(-1)[0]
            #     offset_y4 = (of_y4 * of_valid).max(-1)[0]
            #
            #     offset_ldmks = torch.cat([offset_x0.unsqueeze(-1), offset_y0.unsqueeze(-1),
            #                               offset_x1.unsqueeze(-1), offset_y1.unsqueeze(-1),
            #                               offset_x2.unsqueeze(-1), offset_y2.unsqueeze(-1),
            #                               offset_x3.unsqueeze(-1), offset_y3.unsqueeze(-1),
            #                               offset_x4.unsqueeze(-1), offset_y4.unsqueeze(-1)], dim=-1)
            #     ldmk_targets.append(offset_ldmks.view(-1, 10))
        cls_targets = torch.cat(cls_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        ctr_targets = torch.cat(ctr_targets, dim=0)
        box_mask = torch.cat(box_mask, dim=0)
        if use_ldmks:
            ldmk_targets = torch.cat(ldmk_targets, dim=0)
            if ldmk_reg_type == 'coord':
                ldmk_mask = torch.cat(ldmk_mask, dim=0)
                return box_targets, cls_targets, ctr_targets, box_mask, ldmk_targets, ldmk_mask
            elif ldmk_reg_type == 'heatmap':
                return box_targets, cls_targets, ctr_targets, box_mask, ldmk_targets
        else:
            return box_targets, cls_targets, ctr_targets, box_mask

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


    target_generator = CenterFaceTargetGenerator()
    cls_targets, box_targets, ctr_targets, box_mask, ldmk_targets, ldmk_mask = target_generator.generate_targets(img, boxes, ldmks)

    print(cls_targets.max())
