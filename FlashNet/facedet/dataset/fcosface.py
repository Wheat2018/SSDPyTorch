import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import math
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

WIDER_CLASSES = ('__background__', 'face')

class FCOSFaceDataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, preproc=None, target_transform=None, aug_type='FaceBoxes', use_ldmk=False,
                 ldmk_reg_type=None):
        self.root = root
        self.preproc = preproc
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.aug_type = aug_type
        self.ids = list()
        self.use_ldmk = use_ldmk
        self.ldmk_reg_type = ldmk_reg_type
        if self.use_ldmk:
            img_list = 'trainval.txt'
        else:
            img_list = 'trainval_shuffle.txt'
        for line in open(os.path.join(self.root, 'ImageSets', 'Main', img_list)):
            self.ids.append((self.root, line.strip()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        # print(self._imgpath % img_id)
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            if self.aug_type == 'FaceBoxes':
                img, box_target, box_mask, ldmk_target, ldmk_mask = self.preproc(img, target)
                return torch.from_numpy(img), box_target, box_mask, ldmk_target, ldmk_mask


def detection_collate_fcosface(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    imgs = []
    box_targets = []
    box_masks = []
    ldmk_targets = []
    ldmk_masks = []

    for _, (img, box_target, box_mask, ldmk_target, ldmk_mask) in enumerate(batch):
        if torch.is_tensor(img):
            #             print(img.size())
            imgs.append(img)

        if torch.is_tensor(box_target):
            #             print('box_target',box_target.size())
            box_targets.append(box_target)

        if torch.is_tensor(box_mask):
            #             print('cor_target',cor_target.size())
            box_masks.append(box_mask)

        if torch.is_tensor(ldmk_target):
            # print('ldmk_target',ldmk_target.size())
            ldmk_targets.append(ldmk_target)

        if torch.is_tensor(ldmk_mask):
            # print('ldmk_target',ldmk_target.size())
            ldmk_masks.append(ldmk_mask)

    return (torch.stack(imgs, 0),
            torch.stack(box_targets, 0),
            torch.stack(box_masks, 0),
            torch.stack(ldmk_targets, 0),
            torch.stack(ldmk_masks, 0))