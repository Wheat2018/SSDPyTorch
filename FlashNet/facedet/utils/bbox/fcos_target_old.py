# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os
import sys

import torch
import numpy as np

class FCOSTargetGenerator(object):
    """Generate FCOS targets"""
    # def __init__(self, stages=5, stride=[8, 16, 32, 64, 128],
    #              valid_range=[(0, 40), (40, 90), (90, 160), (160, 320), (320, 1000)], **kwargs):
    # def __init__(self, stages=6, stride=[4, 8, 16, 32, 64, 128],
    #              valid_range=[(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)], **kwargs):
    # def __init__(self, stages=5, stride=[8, 16, 32, 64, 128],
    #              valid_range=[(0, 640), (32, 64), (64, 128), (128, 256), (256, 640)], **kwargs):
    def __init__(self, stages=5, stride=[8, 8, 8, 8, 8],
                 valid_range=[(0, 32), (32, 64), (64, 128), (128, 256), (256, 640)], **kwargs):
        super(FCOSTargetGenerator, self).__init__(**kwargs)
        self._stages = stages
        self._stride = stride
        self._valid_range = valid_range
  
    def generate_targets(self, img, boxes):
        """
        Args:
            img : [H, W, 3]
            boxes : [N, 5]        
        Return:
            cls_targets: [所有步长的特征图点的个数之和, num_classes] 
            ctr_targets: [所有步长的特征图点的个数之和, 1]  
            box_targets: [所有步长的特征图点的个数之和, 4] 
            cor_targets: [所有步长的特征图点的个数之和, 2]  特征图的点对应于原图的坐标       
        """
        _, rh, rw = img.shape
        # print('rh,rw', rh,rw)
        rx = torch.arange(0, rw).view(1, -1)
        ry = torch.arange(0, rh).view(-1, 1)
        sx = rx.repeat(rh, 1).float()
        sy = ry.repeat(1, rw).float()
#         print(boxes)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        areas, inds = torch.sort(areas)
        boxes = boxes[inds]
        boxes = torch.cat([torch.zeros((1, 5)), boxes], dim=0) # for gt assign confusion
        x0, y0, x1, y1, cls = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3],boxes[:,4]
        n = boxes.size(0)

#         # [H, W, N]
        of_l = sx.unsqueeze(2) - x0.unsqueeze(0).unsqueeze(0)
        of_t = sy.unsqueeze(2) - y0.unsqueeze(0).unsqueeze(0)
        of_r = -(sx.unsqueeze(2) - x1.unsqueeze(0).unsqueeze(0))
        of_b = -(sy.unsqueeze(2) - y1.unsqueeze(0).unsqueeze(0))

        # [H, W, N]

        eps = 1e-5
        ctr =(torch.min(of_l, of_r) / torch.max(of_l, of_r)) * \
              (torch.min(of_t, of_b) / torch.max(of_t, of_b))

        ctr = torch.sqrt(torch.abs(ctr))
        ctr[:, :, 0] = 0

        # [H, W, N, 4]
        offsets = torch.cat([of_l.unsqueeze(-1), of_t.unsqueeze(-1),\
                             of_r.unsqueeze(-1), of_b.unsqueeze(-1)], dim=-1)

        if self._stride[0] == 4:
            fh = int(np.ceil(np.ceil(rh / 2) / 2))
            fw = int(np.ceil(np.ceil(rw / 2) / 2))
        elif self._stride[0] == 8:
            fh = int(np.ceil(np.ceil(np.ceil(rh / 2) / 2) / 2))
            fw = int(np.ceil(np.ceil(np.ceil(rw / 2) / 2) / 2))

        fm_list = []
        for i in range(self._stages):
            fm_list.append((fh, fw))
            if self._stride[0] != self._stride[1]:
                fh = int(np.ceil(fh / 2))
                fw = int(np.ceil(fw / 2))
            else:
                fh = int(np.ceil(fh))
                fw = int(np.ceil(fw))
#         fm_list = fm_list[::-1]
#         print('fm_list',fm_list)
        cls_targets = []
        ctr_targets = []
        box_targets = []
        cor_targets = []
    
        for i in range(self._stages):
            fh, fw = fm_list[i]
            cls_target = torch.Tensor(fh, fw)
            box_target = torch.Tensor(fh, fw, 4)
            ctr_target = torch.Tensor(fh, fw)

            rx = torch.arange(0, fw).view(1, -1)
            ry = torch.arange(0, fh).view(-1, 1)
            sx = rx.repeat(fh, 1).float()
            sy = ry.repeat(1, fw).float()
            syx = torch.stack((sy.view(-1), sx.view(-1))).transpose(1,0).long()
            # (fh*fw,)
#             print(syx)
#             import pdb
#             pdb.set_trace()
            by = syx[:, 0] * self._stride[i] + self._stride[i] / 2
            bx = syx[:, 1] * self._stride[i] + self._stride[i] / 2
            by = by.clamp(0, rh - 1)
            bx = bx.clamp(0, rw - 1)
#            by = syx[:, 0] * self._stride[i]
#            bx = syx[:, 1] * self._stride[i]
            # (fh*fw, 2)
            cor_targets.append(torch.stack((bx, by), dim=1)) 

            # [FH*FW, N, 4]
            of_byx = offsets[by, bx]
            min_vr, max_vr = self._valid_range[i]
#             [FH*FW, N]
            # 4个offset都大于零才表示这个点在gt box里
            is_in_box = (torch.prod(of_byx > 0, dim=2)==1)
#            is_valid_area = (of_byx.max(dim=-1)[0] >= min_vr) * (of_byx.max(dim=-1)[0] <= max_vr)
#            is_valid_area=(of_byx[:,:,0]+of_byx[:,:,2]>min_vr)*(of_byx[:,:,1]+of_byx[:,:,3]>min_vr)\
#            *(of_byx[:,:,0]+of_byx[:,:,2]<max_vr)*(of_byx[:,:,1]+of_byx[:,:,3]<max_vr)
            
            box_scale = torch.sqrt((of_byx[:,:,0]+of_byx[:,:,2])*(of_byx[:,:,1]+of_byx[:,:,3]))
#            print('box_scale',box_scale)
            is_valid_area = (box_scale > min_vr) * (box_scale < max_vr)                
            # [FH*FW, N] 
            valid_pos = is_in_box * is_valid_area
            of_valid = torch.zeros((fh, fw, n))
            of_valid[syx[:, 0], syx[:, 1], :] = valid_pos.float() # 1, 0
            of_valid[:, :, 0] = 0
                        
#             import pdb
#             pdb.set_trace()         
  
#             is_in_box = (torch.prod(of_byx > 0, dim=2)==1).unsqueeze(-1).expand_as(of_byx)      
#             is_valid_area = \
#             (torch.prod(of_byx.max(dim=-1)[0] >= min_vr, dim=2)==1).unsqueeze(-1).expand_as(of_byx)\
#             *(torch.prod(of_byx.max(dim=-1)[0] <= max_vr, dim=2)==1).unsqueeze(-1).expand_as(of_byx)
            
#             is_valid_area = (of_byx.max(dim=-1)[0] >= min_vr) * (of_byx.max(dim=-1) <= max_vr)
#             # [FH*FW, N]
#             valid_pos = nd.elemwise_mul(is_in_box, is_valid_area)
#             of_valid = nd.zeros((fh, fw, n))
#             of_valid[syx[:, 0], syx[:, 1], :] = valid_pos # 1, 0
#             of_valid[:, :, 0] = 0
            # [FH, FW]
#             import pdb
#             pdb.set_trace()
            gt_inds = torch.argmax(of_valid, dim=-1)
            gt_inds[torch.argmax(of_valid, dim=-1) == torch.argmin(of_valid, dim=-1)]=0	
            # box targets
            box_target[syx[:, 0], syx[:, 1]] = boxes[gt_inds[syx[:, 0], syx[:, 1]], :4]
            box_target = box_target.view(-1, 4)
            # cls targets
            cls_target[syx[:, 0], syx[:, 1]] = cls[gt_inds[syx[:, 0], syx[:, 1]]]
            cls_target = cls_target.view(-1)          
            
            # ctr targets
            ctr_target[syx[:, 0], syx[:, 1]] = ctr[by, bx, gt_inds[syx[:, 0], syx[:, 1]]]
            ctr_target = ctr_target.view(-1)
            box_targets.append(box_target)
            cls_targets.append(cls_target)
            ctr_targets.append(ctr_target)
#             stride = int(stride / 2)
#         print('box_targets[0].size()',box_targets[0].size())
#         print('box_targets[1].size()',box_targets[1].size())
#         print('box_targets[2].size()',box_targets[2].size())
#         print('box_targets[3].size()',box_targets[3].size())
        box_targets = torch.cat([box_target for box_target in box_targets], dim = 0)
        cls_targets = torch.cat([cls_target for cls_target in cls_targets], dim = 0)

        # ctr_targets = torch.cat([ctr_target for ctr_target in ctr_targets[0:1]], dim = 0)
        ctr_targets = torch.cat([ctr_target for ctr_target in ctr_targets], dim = 0)
        cor_targets = torch.cat([cor_target for cor_target in cor_targets], dim = 0)

#         print('box_targets.size()',box_targets.size())
#         print('cls_targets.size()',cls_targets.size())        
#         print('ctr_targets.size()', ctr_targets.size())
#         print('cor_targets.size()',cor_targets.size())
        
#         print(box_targets)
#         print(cls_targets)        
#         print(ctr_targets)        
#         print(cor_targets)
#         cor_targets = cor_targets.astype('float32')

#         print('box_targets>0',box_targets[box_targets>0].view(-1,4))
#         print('cls_targets>0',cls_targets[cls_targets>0].view(-1,1))        
#         print('ctr_targets>0',ctr_targets[ctr_targets>0].view(-1,1))        
#         print('cor_targets',cor_targets)

        return box_targets, cls_targets, ctr_targets, cor_targets.float()

    def generate_box_cords(self, img_height, img_width):
        """
        Args:
            img : [H, W, 3]
            boxes : [N, 5]        
        Return:
            cls_targets: [所有步长的特征图点的个数之和, num_classes] 
            ctr_targets: [所有步长的特征图点的个数之和, 1]  
            box_targets: [所有步长的特征图点的个数之和, 4] 
            cor_targets: [所有步长的特征图点的个数之和, 2]  特征图的点对应于原图的坐标       
        """
#         _, rh, rw = img.shape
        rh, rw =img_height, img_width
#         print('rh,rw', rh,rw)
        rx = torch.arange(0, rw).view(1, -1)
        ry = torch.arange(0, rh).view(-1, 1)
        sx = rx.repeat(rh, 1).float()
        sy = ry.repeat(1, rw).float()

        fh = int(np.ceil(np.ceil(np.ceil(rh / 2) / 2) / 2))
        fw = int(np.ceil(np.ceil(np.ceil(rw / 2) / 2) / 2))

        fm_list = []
        for i in range(self._stages):
            fm_list.append((fh, fw))
            fh = int(np.ceil(fh / 2))
            fw = int(np.ceil(fw / 2))

        cor_targets = []
    
        for i in range(self._stages):
            fh, fw = fm_list[i]
            rx = torch.arange(0, fw).view(1, -1)
            ry = torch.arange(0, fh).view(-1, 1)
            sx = rx.repeat(fh, 1).float()
            sy = ry.repeat(1, fw).float()
            syx = torch.stack((sy.view(-1), sx.view(-1))).transpose(1,0).long()
            by = syx[:, 0] * self._stride[i] + self._stride[i] / 2
            bx = syx[:, 1] * self._stride[i] + self._stride[i] / 2
#            by = syx[:, 0] * self._stride[i]
#            bx = syx[:, 1] * self._stride[i]
            by = by.clamp(0, rh - 1)
            bx = bx.clamp(0, rw - 1)
            # (fh*fw, 2)
            cor_targets.append(torch.stack((bx, by), dim=1)) 
        cor_targets = torch.cat([cor_target for cor_target in cor_targets], dim = 0)  
        return cor_targets.float()
            
            
def FCOSBoxConverter(box_pred, box_cord):
    """This function is used to convert box_preds(l,t,r,b)
       to corner(x1, y1, x2, y2) format, which then used
       to compute IoULoss.
    """
    # [B, N, 4]->[B, N, 1]
    pl = box_pred[:, :, 0:1]
    pt = box_pred[:, :, 1:2]
    pr = box_pred[:, :, 2:3]
    pb = box_pred[:, :, 3:4]
    # [B, N, 2]->[B, N, 1]
    cx = box_cord[:, :, 0:1]
    cy = box_cord[:, :, 1:2]
    x1 = cx - pl
    y1 = cy - pt
    x2 = cx + pr
    y2 = cy + pb
    boxes = torch.cat([x1, y1, x2, y2], dim=-1)
    return boxes        

def FCOSBoxTargetConverter(box_target, box_cord):
    """This function is used to convert box_targets(x1, y1, x2, y2)
       to corner(l,t,r,b) format, which then used
       to compute IoULoss.
    """
    # [B, N, 4]->[B, N, 1]
    x1 = box_target[:, :, 0:1]
    y1 = box_target[:, :, 1:2]
    x2 = box_target[:, :, 2:3]
    y2 = box_target[:, :, 3:4]

    # [B, N, 2]->[B, N, 1]
    cx = box_cord[:, :, 0:1]
    cy = box_cord[:, :, 1:2]
    pl = cx - x1
    pt = cy - y1
    pr = x2 - cx
    pb = y2 - cy

    box_targets = torch.cat([pl, pt, pr, pb], dim=-1)
    return box_targets

#     def compute_locations(self, features):
#         locations = []
#         for level, feature in enumerate(features):
#             h, w = feature.size()[-2:]
#             locations_per_level = self.compute_locations_per_level(
#                 h, w, self.fpn_strides[level],
#                 feature.device
#             )
#             locations.append(locations_per_level)
#         return locations

#     def compute_locations_per_level(self, h, w, stride, device):
#         shifts_x = torch.arange(
#             0, w * stride, step=stride,
#             dtype=torch.float32, device=device
#         )
#         shifts_y = torch.arange(
#             0, h * stride, step=stride,
#             dtype=torch.float32, device=device
#         )
#         shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#         shift_x = shift_x.reshape(-1)
#         shift_y = shift_y.reshape(-1)
#         locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
#         return locations
if __name__ == '__main__':
    img = torch.zeros(3, 600, 899) #hwc

    boxes = torch.FloatTensor([[1, 88, 821, 480, 60],
                      [1, 75, 600, 251, 60],
                      [1, 235, 899, 598, 60],
                      [336, 48, 395, 117, 29]])

    target_generator = FCOSTargetGenerator()
    cls_targets, ctr_targets, box_targets, cor_targets = target_generator.generate_targets(img, boxes)
    
    