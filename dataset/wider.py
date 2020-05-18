"""
    By.Wheat
    2020.05.17
"""

import torch.utils.data as data
from common import *

WIDER_ROOT = path.join(DATA_ROOT, 'WIDER')

class WIDER(data.Dataset):
    image_folds = {
        'train': ['train'],
        'val': ['val'],
        'test': ['test'],
        'train_val': ['train', 'val'],
        'all': ['train', 'val', 'test']
    }

    cfg = {
        'num_classes': 2,
        'lr_steps': (20000, 40000, 60000, 80000),
        'max_iter': 120000,
        'epochs': 20,
    }

    def __init__(self,
                 root=WIDER_ROOT,
                 image_list_txt='wider_face_%s_bbx_gt.txt',
                 dataset='train',           # train or val or test or all
                 image_enhancement_fn=None
                 ):
        self.root = root
        self.image_list_txt = image_list_txt
        self.dataset = dataset
        self.image_enhancement_fn = image_enhancement_fn

        self.image_names = []
        self.image_rawBoxes = []
        self.image_preBoxes = []
        self.range_of_each_fold = dict()
        self.name = type(self).__name__

        for fold in self.image_folds[dataset]:
            start_index = len(self.image_names)
            if fold == 'test':
                f = open(path.join(root, 'wider_face_split', 'wider_face_test_filelist.txt'))
                line = f.readline()
                while line:
                    self.image_names.append(line.replace("\n", ""))
                    self.image_rawBoxes.append([])
                    line = f.readline()
            else:
                f = open(path.join(root, 'wider_face_split', image_list_txt % fold))
                line = f.readline()
                while line:
                    self.image_names.append(line.replace("\n", ""))
                    line = f.readline()
                    num_of_boxes = int(line)
                    if num_of_boxes == 0:       # In WIDER, images without boxes followed by '0 0 0 0 0 0 0 0 0 0' line.
                        f.readline()
                        line = f.readline()
                        continue
                    raw_boxes = []
                    for i in range(num_of_boxes):
                        line = f.readline()
                        box_ele = line.split(' ')
                        raw_box = [float(box_ele[0]),
                                   float(box_ele[1]),
                                   float(box_ele[0]) + float(box_ele[2]),
                                   float(box_ele[1]) + float(box_ele[3])]
                        raw_boxes.append(raw_box)
                    self.image_rawBoxes.append(raw_boxes)
                    line = f.readline()
            end_index = len(self.image_names)
            self.range_of_each_fold[fold] = [start_index, end_index]
        self.image_preBoxes = [torch.Tensor() for i in range(len(self))]

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.image_names)

    def pull_item(self, index):
        image = self.pull_image(index)
        assert image is not None
        h, w, channels = image.shape
        boxes_classes = []
        for rawBox in self.image_rawBoxes[index]:
            rawBox[0] /= w
            rawBox[1] /= h
            rawBox[2] /= w
            rawBox[3] /= h
            rawBox += [0.]
            boxes_classes.append(rawBox)
        if self.image_enhancement_fn is not None:
            boxes_classes = np.array(boxes_classes)
            if len(boxes_classes) == 0:
                image, _, _ = self.image_enhancement_fn(image, None, None)
            else:
                image, boxes, classes = self.image_enhancement_fn(image, boxes_classes[:, :4], boxes_classes[:, 4])
                # to rgb
                image = image[:, :, (2, 1, 0)]
                boxes_classes = np.hstack((boxes, np.expand_dims(classes, axis=1)))
        return torch.from_numpy(image).permute(2, 0, 1), boxes_classes, h, w

    def pull_image(self, index):
        return cv2.imread(path.join(self.root, 'images', self.image_names[index]))

    def pull_anno(self, index):
        image_name = self.image_names[index]
        pass

    def pull_tensor(self, index):
        pass


if __name__ == '__main__':
    temp = WIDER(dataset='test')
    img, gt_boxes, height, width = temp.pull_item(1)
    print(gt_boxes)
