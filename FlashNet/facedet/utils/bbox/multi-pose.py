from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from .image import flip, color_aug
from .image import get_affine_transform, affine_transform
from .image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from .image import draw_dense_reg
from facedet.utils.utils import Data_anchor_sample
from facedet.utils.Randaugmentations import Randaugment
import math
from PIL import Image
import re
from torch._six import container_abcs, string_classes, int_classes

np_str_obj_array_pattern = re.compile(r'[SaUO]')

class MultiPoseDataset(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = len(anns)
    # num_objs = min(len(anns), self.max_objs)
    if num_objs > self.max_objs:
        num_objs = self.max_objs
        anns = np.random.choice(anns, num_objs)

    img = cv2.imread(img_path)

    img, anns = Data_anchor_sample(img, anns)

    # # for test the keypoint order
    # img1 = cv2.flip(img,1)
    # for ann in anns:
    #   width = img1.shape[1]
    #   bbox = self._coco_box_to_bbox(ann['bbox'])
    #   bbox[[0, 2]] = width - bbox[[2, 0]] - 1
    #   pts = np.array(ann['keypoints'], np.float32).reshape(5, 3)
    #
    #   # for flip
    #   pts[:, 0] = width - pts[:, 0] - 1
    #   for e in self.flip_idx:
    #     pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
    #
    #   # for debug show
    #   def add_coco_bbox(image, bbox, conf=1):
    #     txt = '{}{:.1f}'.format('person', conf)
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
    #     cv2.putText(image, txt, (bbox[0], bbox[1] - 2),
    #                 font, 0.5, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    #
    #   def add_coco_hp(image, points, keypoints_prob=1):
    #     for j in range(5):
    #       if keypoints_prob > 0.5:
    #         if j == 0:
    #           cv2.circle(image, (points[j, 0], points[j, 1]), 2, (255, 255, 0), -1)
    #         elif j == 1:
    #           cv2.circle(image, (points[j, 0], points[j, 1]), 2, (255, 0, 0), -1)
    #         elif j == 2:
    #           cv2.circle(image, (points[j, 0], points[j, 1]), 2, (0, 255, 0), -1)
    #         elif j == 3:
    #           cv2.circle(image, (points[j, 0], points[j, 1]), 2, (0, 0, 255), -1)
    #         elif j == 4:
    #           cv2.circle(image, (points[j, 0], points[j, 1]), 2, (0, 0, 0), -1)
    #     return image
    #
    #   bbox = [int(x) for x in bbox]
    #   add_coco_bbox(img1, bbox )
    #   add_coco_hp(img1, pts)
    #   cv2.imshow('mat', img1)
    #   cv2.waitKey(5000)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    s = max(img.shape[0], img.shape[1]) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        # s = s * np.random.choice(np.arange(0.8, 1.1, 0.1))
        s = s
        # _border = np.random.randint(128*0.4, 128*1.4)
        _border = s * np.random.choice([0.1, 0.2, 0.25])
        w_border = self._get_border(_border, img.shape[1])
        h_border = self._get_border(_border, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      if np.random.random() < self.opt.aug_rot:
        rf = self.opt.rotate
        rot = np.clip(np.random.randn()*rf, -rf*2, rf*2)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] = width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input,
                         (self.opt.input_res, self.opt.input_res),
                         flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:                 # 随机进行图片增强
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      # inp = Randaugment(self._data_rng, inp, self._eig_val, self._eig_vec)

    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    num_joints = self.num_joints
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((num_joints, output_res, output_res), dtype=np.float32)
    dense_kps = np.zeros((num_joints, 2, output_res, output_res),
                          dtype=np.float32)
    dense_kps_mask = np.zeros((num_joints, output_res, output_res),
                               dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    wight_mask = np.ones((self.max_objs), dtype=np.float32)
    kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      cls_id = int(ann['category_id']) - 1
      pts = np.array(ann['keypoints'], np.float32).reshape(num_joints, 3)
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        pts[:, 0] = width - pts[:, 0] - 1
        for e in self.flip_idx:
          pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()

      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox = np.clip(bbox, 0, output_res - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if (h > 0 and w > 0) or (rot != 0):
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)       # 人脸的中心坐标
        ct_int = ct.astype(np.int32)                        # 整数化
        # wh[k] = 1. * w, 1. * h                                                    # 2. centernet的方式
        wh[k] = np.log(1. * w / 4), np.log(1. * h / 4)                              # 2. 人脸bbox的高度和宽度,centerface论文的方式
        ind[k] = ct_int[1] * output_res + ct_int[0]         # 人脸bbox在1/4特征图中的索引
        reg[k] = ct - ct_int                                # 3. 人脸bbox中心点整数化的偏差
        reg_mask[k] = 1                                     # 是否需要用于计算误差
        # if w*h <= 20:
        #     wight_mask[k] = 15

        num_kpts = pts[:, 2].sum()                           # 没有关键点标注的时哦
        if num_kpts == 0:                                    # 没有关键点标注的都是比较困难的样本
          # print('没有关键点标注')
          hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
          # reg_mask[k] = 0

        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius))
        for j in range(num_joints):
          if pts[j, 2] > 0:
            pts[j, :2] = affine_transform(pts[j, :2], trans_output_rot)
            if pts[j, 0] >= 0 and pts[j, 0] < output_res and \
               pts[j, 1] >= 0 and pts[j, 1] < output_res:
              kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int                # 4. 关键点相对于人脸bbox的中心的偏差
              kps_mask[k, j * 2: j * 2 + 2] = 1
              pt_int = pts[j, :2].astype(np.int32)                          # 关键点整数化
              hp_offset[k * num_joints + j] = pts[j, :2] - pt_int           # 关键点整数化的偏差
              hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]   # 索引
              hp_mask[k * num_joints + j] = 1                                   # 计算损失的mask
              if self.opt.dense_hp:
                # must be before draw center hm gaussian
                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                               pts[j, :2] - ct_int, radius, is_offset=True)
                draw_gaussian(dense_kps_mask[j], ct_int, radius)
              draw_gaussian(hm_hp[j], pt_int, hp_radius)                    # 1. 关键点高斯map
              if ann['bbox'][2]*ann['bbox'][3] <= 16.0:                   # 太小的人脸忽略
                kps_mask[k, j * 2: j * 2 + 2] = 0
        draw_gaussian(hm[cls_id], ct_int, radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1] +
                       pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
    if rot != 0:
      hm = hm * 0 + 0.9999
      reg_mask *= 0
      kps_mask *= 0
    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
           'landmarks': kps, 'hps_mask': kps_mask, 'wight_mask': wight_mask}
    if self.opt.dense_hp:
      dense_kps = dense_kps.reshape(num_joints * 2, output_res, output_res)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints, 1, output_res, output_res)
      dense_kps_mask = np.concatenate([dense_kps_mask, dense_kps_mask], axis=1)
      dense_kps_mask = dense_kps_mask.reshape(
        num_joints * 2, output_res, output_res)
      ret.update({'dense_hps': dense_kps, 'dense_hps_mask': dense_kps_mask})
      del ret['hps'], ret['hps_mask']
    if self.opt.reg_offset:
      ret.update({'hm_offset': reg})                  # 人脸bbox中心点整数化的偏差
    if self.opt.hm_hp:
      ret.update({'hm_hp': hm_hp})
    if self.opt.reg_hp_offset:
      ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 40), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    return ret


_use_shared_memory = False

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))


def multipose_collate(batch):
  objects_dims = [d.shape[0] for d in batch]
  index = objects_dims.index(max(objects_dims))

  # one_dim = True if len(batch[0].shape) == 1 else False
  res = []
  for i in range(len(batch)):
      tres = np.zeros_like(batch[index], dtype=batch[index].dtype)
      tres[:batch[i].shape[0]] = batch[i]
      res.append(tres)

  return res


def Multiposebatch(batch):
  sample_batch = {}
  for key in batch[0]:
    if key in ['hm', 'input']:
      sample_batch[key] = default_collate([d[key] for d in batch])
    else:
      align_batch = multipose_collate([d[key] for d in batch])
      sample_batch[key] = default_collate(align_batch)

  return sample_batch


import time
from .image import transform_preds

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # 前100个点

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs

def centerface_decode(
        heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
    '''
    :param heat: bchw
    :param wh:
    :param kps:
    :param reg:
    :param hm_hp:
    :param hp_offset:
    :param K:
    :return:
    '''
    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys_int, xs_int = _topk(heat, K=K)
    # scores, inds, clses, ys_int, xs_int, K = threshold_choose(heat, threshold=0.05)

    if reg is not None:  # 回归的中心点偏移量
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs_int.view(batch, K, 1) + reg[:, :, 0:1]                  # 1. 中心点，后面乘了4
        ys = ys_int.view(batch, K, 1) + reg[:, :, 1:2]
        # xs = (xs_int.view(batch, K, 1) + reg[:, :, 0:1] + 0.5)
        # ys = (ys_int.view(batch, K, 1) + reg[:, :, 1:2] + 0.5)            # 1. 中心点，按centerface的方式计算
    else:
        xs = xs_int.view(batch, K, 1) + 0.5
        ys = ys_int.view(batch, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)  # 人脸bbox矩形框的宽高
    wh = wh.view(batch, K, 2)                                             # 2. wh,第一种方式
    wh = wh.exp() * 4.                                                    # 2. wh,第二种式式
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)

    kps = _tranpose_and_gather_feat(kps, inds)                                      # 3. 人脸关键点
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)  # 第一次通过中心点偏移获得的关节点的坐标
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)  # 第二次：通过关节点热力图求得关节点的中心点
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:  # 关节点的中心的偏移
            hp_offset = _tranpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()  # 选置信度大于0.1的
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)  # 两次求解的关节点求距离
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)

        # 如果在bboxes中则用第二种方法的关节点，在bboxes外用第一种方法提取的关节点，就是优先选第二种方法
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
    detections = torch.cat([bboxes, scores, kps, clses], dim=2)  # box:4+score:1+kpoints:10+class:1=16

    return detections

def flip_tensor(x):
    return torch.flip(x, [3])

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2,
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)



def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      # inp_height = (new_height | self.opt.pad) + 1
      # inp_width = (new_width | self.opt.pad) + 1
      inp_height = int(np.ceil(new_height / 32) * 32)
      inp_width = int(np.ceil(new_width / 32) * 32)
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // self.opt.down_ratio,
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

def process(self, images, return_time=False):
    with torch.no_grad():
        torch.cuda.synchronize()
        output = self.model(images)[-1]
        output['hm'] = output['hm']
        # if self.opt.hm_hp and not self.opt.mse_loss:
        #   output['hm_hp'] = output['hm_hp'].sigmoid_()

        reg = output['hm_offset'] if self.opt.reg_offset else None
        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        torch.cuda.synchronize()
        forward_time = time.time()

        if self.opt.flip_test:
            output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
            output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
            output['hps'] = (output['hps'][0:1] +
                             flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
            hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                if hm_hp is not None else None
            reg = reg[0:1] if reg is not None else None
            hp_offset = hp_offset[0:1] if hp_offset is not None else None

        dets = centerface_decode(
            output['hm'], output['wh'], output['landmarks'],
            reg=reg, K=self.opt.K)

    if return_time:
        return output, dets, forward_time
    else:
        return output, dets

def multi_pose_post_process(dets, c, s, h, w):
  # dets的数据格式为：box: 4 + score:1 + kpoints: 10 +  class: 1 = 16
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))         # 矩形框
    pts = transform_preds(dets[i, :, 5:15].reshape(-1, 2), c[i], s[i], (w, h))        # 1-关键点数
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],                                          # 置信度
       pts.reshape(-1, 10)], axis=1).astype(np.float32).tolist()                      # 2-关键点数×2
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret

def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets = multi_pose_post_process(
      dets.copy(), [meta['c']], [meta['s']],
      meta['out_height'], meta['out_width'])
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 15)             # 关键点数+5=15
      # import pdb; pdb.set_trace()
      dets[0][j][:, :4] /= scale
      dets[0][j][:, 5:] /= scale
    return dets[0]

def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
        image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type(''):
        image = cv2.imread(image_or_path_or_tensor)
    else:
        image = image_or_path_or_tensor['image'][0].numpy()
        pre_processed_images = image_or_path_or_tensor
        pre_processed = True

    loaded_time = time.time()
    load_time += (loaded_time - start_time)

    detections = []
    for scale in self.scales:
        scale_start_time = time.time()
        if not pre_processed:
            images, meta = self.pre_process(image, scale, meta)
        else:
            # import pdb; pdb.set_trace()
            images = pre_processed_images['images'][scale][0]
            meta = pre_processed_images['meta'][scale]
            meta = {k: v.numpy()[0] for k, v in meta.items()}
        images = images.to(self.opt.device)
        torch.cuda.synchronize()
        pre_process_time = time.time()
        pre_time += pre_process_time - scale_start_time

        output, dets, forward_time = self.process(images, return_time=True)

        torch.cuda.synchronize()
        net_time += forward_time - pre_process_time
        decode_time = time.time()
        dec_time += decode_time - forward_time

        if self.opt.debug >= 2:
            self.debug(debugger, images, dets, output, scale)

        dets = self.post_process(dets, meta, scale)  # box:4+score:1+kpoints:10+class:1=16
        torch.cuda.synchronize()
        post_process_time = time.time()
        post_time += post_process_time - decode_time

        detections.append(dets)

    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
        self.show_results(debugger, image, results)

    if self.opt.debug == -1:
        plot_img = self.return_results(debugger, image, results)
    else:
        plot_img = None

    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time, 'plot_img': plot_img}


class MultiPoseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MultiPoseLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
            torch.nn.L1Loss(reduction='sum')
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        lm_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            output['hm'] = output['hm']
            # if opt.hm_hp and not opt.mse_loss:
            #   output['hm_hp'] = _sigmoid(output['hm_hp'])

            if opt.eval_oracle_hmhp:
                output['hm_hp'] = batch['hm_hp']
            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_kps:
                if opt.dense_hp:
                    output['hps'] = batch['dense_hps']
                else:
                    output['hps'] = torch.from_numpy(gen_oracle_map(
                        batch['hps'].detach().cpu().numpy(),
                        batch['ind'].detach().cpu().numpy(),
                        opt.output_res, opt.output_res)).to(opt.device)
            if opt.eval_oracle_hp_offset:
                output['hp_offset'] = torch.from_numpy(gen_oracle_map(
                    batch['hp_offset'].detach().cpu().numpy(),
                    batch['hp_ind'].detach().cpu().numpy(),
                    opt.output_res, opt.output_res)).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks  # 1. focal loss,求目标的中心，
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],  # 2. 人脸bbox高度和宽度的loss
                                         batch['ind'], batch['wh'], batch['wight_mask']) / opt.num_stacks
            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['hm_offset'], batch['reg_mask'],  # 3. 人脸bbox中心点下采样，所需要的偏差补偿
                                          batch['ind'], batch['hm_offset'], batch['wight_mask']) / opt.num_stacks

            if opt.dense_hp:
                mask_weight = batch['dense_hps_mask'].sum() + 1e-4
                lm_loss += (self.crit_kp(output['hps'] * batch['dense_hps_mask'],
                                         batch['dense_hps'] * batch['dense_hps_mask']) /
                            mask_weight) / opt.num_stacks
            else:
                lm_loss += self.crit_kp(output['landmarks'], batch['hps_mask'],  # 4. 关节点的偏移
                                        batch['ind'], batch['landmarks']) / opt.num_stacks

            # if opt.reg_hp_offset and opt.off_weight > 0:                              # 关节点的中心偏移
            #   hp_offset_loss += self.crit_reg(
            #     output['hp_offset'], batch['hp_mask'],
            #     batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
            # if opt.hm_hp and opt.hm_hp_weight > 0:                                    # 关节点的热力图
            #   hm_hp_loss += self.crit_hm_hp(
            #     output['hm_hp'], batch['hm_hp']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.lm_weight * lm_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'lm_loss': lm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats