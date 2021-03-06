import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


VEHICLE_CLASSES = ( '__background__', 'Vehicle')


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True, with_landmark=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VEHICLE_CLASSES, range(len(VEHICLE_CLASSES))))
        self.keep_difficult = keep_difficult
        self.with_landmark = with_landmark

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            # name = obj.find('name').text.lower().strip()
            name = obj.find('name').text.strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res



class VOCDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """
    
    def __init__(self, root, preproc=None, target_transform=None, is_anchor_free = False, is_ctr_target=False, aug_type = 'FaceBoxes'):
        self.root = root
        self.preproc = preproc
        self.target_transform = target_transform
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._is_anchor_free = is_anchor_free
        self._is_ctr_target = is_ctr_target
        self.aug_type = aug_type
        self.ids = list()

        for line in open(os.path.join(self.root, 'ImageSets', 'Main', 'trainval.txt')):
            self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        img_id = self.ids[index]
#         print(img_id)
#         print('anno',self._annopath % img_id)
#         print('img',self._imgpath % img_id)
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        # forward
        # img_origin = img.copy()
        # # import numpy as np
        # # img = img.transpose(2, 0, 1).clip(0, 255)
        # boxes = target[:, 0:4].reshape(-1,4)
        #
        # boxes = boxes
        # for box in boxes:
        #     xmin, ymin, xmax, ymax = np.ceil(box)
        #     cv2.rectangle(img_origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2)
        #
        # points = target[:,5:].reshape(-1,2)
        # points = points
        # for point in points:
        #     # print('point',(int(point[0]), int(point[1])))
        #     if point[0] > 0:
        #         cv2.circle(img_origin, (int(point[0]), int(point[1])), radius=1, color=(155, 100, 255), thickness=2)
        # print(img_id[1])
        # # print(os.path.join('./show_train_img_with_anno/', img_id[1].split('/')[0]))
        # if not os.path.exists(os.path.join('./show_train_img_with_anno/', img_id[1].split('/')[0])):
        #     os.makedirs(os.path.join('./show_train_img_with_anno/', img_id[1].split('/')[0]))
        #
        # cv2.imwrite(os.path.join('./show_train_img_with_anno/', img_id[1] + '.jpg'), img_origin)


        if self.preproc is not None:
#             print('self.preproc')
            if self.aug_type == 'FaceBoxes':
#                 print('self.preproc faceboxes')
                if self._is_anchor_free:
                    img, box_target, cls_target, ctr_target, cor_target = self.preproc(img, target)
                elif self._is_ctr_target:

                    img, target, ctr_target = self.preproc(img, target)
                else:
                    img, target = self.preproc(img, target)

                    # aug_img = img.transpose(1, 2, 0).copy()
                    # aug_img += (104, 117, 123)
                    # aug_img = aug_img.astype(np.uint8)
                    #
                    # boxes = target[:, 0:4].reshape(-1, 4)
                    # boxes = boxes*640
                    # for box in boxes:
                    #     xmin, ymin, xmax, ymax = np.ceil(box)
                    #     cv2.rectangle(aug_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2)
                    #
                    # points = target[:, 5:].reshape(-1, 2)
                    # points = points*640
                    # for point in points:
                    #     # print('point',(int(point[0]), int(point[1])))
                    #     if point[0] > 0:
                    #         cv2.circle(aug_img, (int(point[0]), int(point[1])), radius=1, color=(155, 100, 255), thickness=1)
                    # print(os.path.join('./show_train_img_with_anno/', img_id[1].split('/')[0]))
                    # if not os.path.exists(os.path.join('./show_train_img_with_anno/', img_id[1].split('/')[0])):
                    #     os.makedirs(os.path.join('./show_train_img_with_anno/', img_id[1].split('/')[0]))
                    #
                    # cv2.imwrite(os.path.join('./show_train_img_with_anno/', img_id[1]  + '_after.jpg'), aug_img)

            elif self.aug_type == 'DSFD': 
                if self._is_anchor_free:
                    raise NOTIMPLENTERROR
                else:
                    img, target = self.preproc(img, target[:, :4], target[:, 4])
                
            if self._is_anchor_free:
                return torch.from_numpy(img), box_target, cls_target, ctr_target, cor_target
            elif self._is_ctr_target:
                # print('ctr target true')
                # print(torch.from_numpy(target).size())
                # print(ctr_target.size())
                return torch.from_numpy(img), torch.from_numpy(target).float(), ctr_target
            else:
                return torch.from_numpy(img), target


    def __len__(self):
        return len(self.ids)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
#                 print(tup.size())
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


def detection_collate_ctr(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    ctr_targets = []
    imgs = []
    for _, (img, target, ctr_target) in enumerate(batch):
        if torch.is_tensor(img):
            imgs.append(img)
        if torch.is_tensor(target):
            targets.append(target)
        if torch.is_tensor(ctr_target):
            ctr_targets.append(ctr_target)

    return (torch.stack(imgs, 0), targets, torch.stack(ctr_targets,0))


def detection_collate_fcos(batch):
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
    cls_targets = []
    ctr_targets = []
    cor_targets = []
    
    for _, (img, box_target, cls_target, ctr_target, cor_target) in enumerate(batch):
#         print(sample[0].size())
#         print(sample[1].size())
#         print(sample[2].size())
#         print(sample[3].size())
#         for _, (img, cls_target, box_target, ctr_target) in enumerate(sample):
        if torch.is_tensor(img):
#             print(img.size())
            imgs.append(img)

        if torch.is_tensor(cls_target):
#             print('cls_target',cls_target.size())
            cls_targets.append(cls_target)

        if torch.is_tensor(box_target):
#             print('box_target',box_target.size())
            box_targets.append(box_target)

        if torch.is_tensor(ctr_target):
#             print('ctr_target',ctr_target.size())
            ctr_targets.append(ctr_target)

        if torch.is_tensor(cor_target):
#             print('cor_target',cor_target.size())
            cor_targets.append(cor_target)
    
    return (torch.stack(imgs, 0), torch.stack(box_targets, 0), \
            torch.stack(cls_targets, 0).unsqueeze(-1), \
            torch.stack(ctr_targets, 0).unsqueeze(-1), torch.stack(cor_targets, 0))







