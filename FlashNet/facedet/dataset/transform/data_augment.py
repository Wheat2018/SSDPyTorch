from .img_tranforms import *
from ....facedet.utils.bbox.fcos_target import FCOSTargetGenerator
import torch

# target_generator = FCOSTargetGenerator()
target_generator = FCOSTargetGenerator(stages=2, stride=[64, 128], valid_range=[(384, 480), (480, 960)])

class preproc(object):

    def __init__(self, img_dim, rgb_means, is_anchor_free=False, is_ctr_target=False, is_xface=False, **kwargs):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.is_anchor_free = is_anchor_free
        self.is_ctr_target = is_ctr_target
        self.is_xface = is_xface

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        # print(image.shape)

        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # image_t = _distort(image)
        # image_t, boxes_t = _expand(image_t, boxes, self.cfg['rgb_mean'], self.cfg['max_expand_ratio'])
        # image_t, boxes_t, labels_t = _crop(image_t, boxes, labels, self.img_dim, self.rgb_means)
        image_t, boxes_t, labels_t, pad_image_flag = crop(image, boxes, labels, self.img_dim)
        image_t = distort(image_t)
        image_t = pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t = mirror(image_t, boxes_t)
        height, width, _ = image_t.shape
        # print('image_t.shape',image_t.shape)
        image_t = resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        if self.is_xface:
            # print('width, height', width, height)
            # print('image_t.shape', image_t.shape)
            # import pdb
            # pdb.set_trace()
            boxes_t[:, 0::2] /= width
            boxes_t[:, 1::2] /= height
            labels_t = np.expand_dims(labels_t, 1)
            anchor_base_targets_t = np.hstack((boxes_t, labels_t))

            # labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape
            # boxes_t[:, 0::2] /= (width / new_width)
            # boxes_t[:, 1::2] /= (height / new_height)
            boxes_t[:, 0::2] *= new_width
            boxes_t[:, 1::2] *= new_height
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            anchor_free_box_target, anchor_free_cls_target, ctr_target, cor_target = \
            target_generator.generate_targets(image_t, boxes)
            return image_t, anchor_base_targets_t, anchor_free_box_target, anchor_free_cls_target, ctr_target, cor_target

        elif self.is_anchor_free:
            labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape
            boxes_t[:, 0::2] /= (width / new_width)
            boxes_t[:, 1::2] /= (height / new_height)
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            box_target, cls_target, ctr_target, cor_target = \
            target_generator.generate_targets(image_t, boxes) 
            return image_t, box_target, cls_target, ctr_target, cor_target

        elif self.is_ctr_target:
            labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape

            # print('width / new_width', width / new_width)
            # print('height / new_height', height / new_height)

            boxes_t[:, 0::2] /= (width / new_width)
            boxes_t[:, 1::2] /= (height / new_height)
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            _, _, ctr_target, _ = \
                target_generator.generate_targets(image_t, boxes)

            boxes_t[:, 0::2] /= (new_width)
            boxes_t[:, 1::2] /= (new_height)
            targets_t = np.hstack((boxes_t, labels_t))

            return image_t, targets_t, ctr_target

        else:        
            boxes_t[:, 0::2] /= width
            boxes_t[:, 1::2] /= height

            labels_t = np.expand_dims(labels_t, 1)
            targets_t = np.hstack((boxes_t, labels_t))
            return image_t, targets_t

class finetune_preproc(object):

    def __init__(self, img_dim, rgb_means, is_anchor_free=False, is_ctr_target=False, is_xface=False, **kwargs):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.is_anchor_free = is_anchor_free
        self.is_ctr_target = is_ctr_target
        self.is_xface = is_xface

    def __call__(self, image, targets):
        print('finetune_preproc', targets.shape)
        # assert targets.shape[0] > 0, "this image does not have gt"
        # print(image.shape)

        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # image_t = _distort(image)
        # image_t, boxes_t = _expand(image_t, boxes, self.cfg['rgb_mean'], self.cfg['max_expand_ratio'])
        # image_t, boxes_t, labels_t = _crop(image_t, boxes, labels, self.img_dim, self.rgb_means)
        # image_t, boxes_t, labels_t, pad_image_flag = _finetune_crop(image, boxes, labels, self.img_dim)
        image_t, boxes_t, labels_t, pad_image_flag = image, boxes, labels, True
        image_t = distort(image_t)
        image_t = pad_to_square(image_t,self.rgb_means, pad_image_flag)
        image_t, boxes_t = mirror(image_t, boxes_t)
        height, width, _ = image_t.shape
        # print('image_t.shape',image_t.shape)
        image_t = resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

        if self.is_xface:
            # print('width, height', width, height)
            # print('image_t.shape', image_t.shape)
            # import pdb
            # pdb.set_trace()
            boxes_t[:, 0::2] /= width
            boxes_t[:, 1::2] /= height
            labels_t = np.expand_dims(labels_t, 1)
            anchor_base_targets_t = np.hstack((boxes_t, labels_t))

            # labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape
            # boxes_t[:, 0::2] /= (width / new_width)
            # boxes_t[:, 1::2] /= (height / new_height)
            boxes_t[:, 0::2] *= new_width
            boxes_t[:, 1::2] *= new_height
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            anchor_free_box_target, anchor_free_cls_target, ctr_target, cor_target = \
            target_generator.generate_targets(image_t, boxes)
            return image_t, anchor_base_targets_t, anchor_free_box_target, anchor_free_cls_target, ctr_target, cor_target

        elif self.is_anchor_free:
            labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape
            boxes_t[:, 0::2] /= (width / new_width)
            boxes_t[:, 1::2] /= (height / new_height)
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            box_target, cls_target, ctr_target, cor_target = \
            target_generator.generate_targets(image_t, boxes)
            return image_t, box_target, cls_target, ctr_target, cor_target

        elif self.is_ctr_target:
            labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape

            # print('width / new_width', width / new_width)
            # print('height / new_height', height / new_height)

            boxes_t[:, 0::2] /= (width / new_width)
            boxes_t[:, 1::2] /= (height / new_height)
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            _, _, ctr_target, _ = \
                target_generator.generate_targets(image_t, boxes)

            boxes_t[:, 0::2] /= (new_width)
            boxes_t[:, 1::2] /= (new_height)
            targets_t = np.hstack((boxes_t, labels_t))

            return image_t, targets_t, ctr_target

        else:
            boxes_t[:, 0::2] /= width
            boxes_t[:, 1::2] /= height

            labels_t = np.expand_dims(labels_t, 1)
            targets_t = np.hstack((boxes_t, labels_t))
            return image_t, targets_t