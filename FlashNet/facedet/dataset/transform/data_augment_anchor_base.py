from ....facedet.utils.bbox.fcos_target_old import FCOSTargetGenerator
import torch
import numpy as np
from .img_tranforms import *

target_generator = FCOSTargetGenerator()

class preproc_ldmk(object):

    def __init__(self, img_dim, rgb_means, is_anchor_free=False, is_ctr_target=False, **kwargs):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.is_anchor_free = is_anchor_free
        self.is_ctr_target = is_ctr_target

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        # import pdb
        # pdb.set_trace()
        ldmks = targets[:, [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]].copy()
        # ldmks = targets[:, 5:].copy()
        image_t, boxes_t, labels_t, ldmks_t, pad_image_flag = crop_with_ldmk(image, boxes, labels, ldmks, self.img_dim)
        image_t = distort(image_t)
        image_t = pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, ldmks_t = mirror_with_ldmk(image_t, boxes_t, ldmks_t)
        height, width, _ = image_t.shape
        # print('image_t.shape',image_t.shape)
        image_t = resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height
        # print(boxes_t)
        mask_landmarks = (ldmks_t == -1).all(axis=1)
        ldmks_t[:, 0::2] /= width
        ldmks_t[:, 1::2] /= height
        ldmks_t[mask_landmarks] = -1.0
        # print(landmarks_t)
        # print('before', landmarks_t[mask_landmarks])
        # print('before', landmarks_t[mask_landmarks][:, 0::2])
        # print('width', width)
        # print(landmarks_t[mask_landmarks][:, 0::2]/width)
        # landmarks_t[mask_landmarks][:, 0::2] = (landmarks_t[mask_landmarks][:, 0::2]/width)
        # print('after', landmarks_t[mask_landmarks][:, 0::2])
        # landmarks_t[mask_landmarks][:, 1::2] /= height
        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t, ldmks_t))
        return image_t, targets_t