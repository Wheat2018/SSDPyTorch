"""
    By.Wheat
    2020.05.11
"""

import torch.utils.data as data
from common import *

FDDB_ROOT = path.join(DATA_ROOT, 'FDDB')


def bounding_rect_of_eclipse(major_axis_radius,
                             minor_axis_radius,
                             angle,
                             center_x,
                             center_y):
    """
    :param major_axis_radius: major semi axis
    :param minor_axis_radius: minor semi axis
    :param angle: anticlockwise rotation angle
    :param center_x:
    :param center_y:
    standard expression of eclipse
        A*x^2 + B*xy + C*y^2 + F = 0
    :return: [left, top, right, bottom]
    """
    cos_angle = math.cos(angle/180.*math.pi)
    sin_angle = math.sin(angle/180.*math.pi)
    a = major_axis_radius**2 * cos_angle**2 + minor_axis_radius**2 * sin_angle**2
    b = 2 * sin_angle * cos_angle * (major_axis_radius**2 - minor_axis_radius**2)
    c = major_axis_radius**2 * sin_angle**2 + minor_axis_radius**2 * cos_angle**2
    f = -major_axis_radius**2 * minor_axis_radius**2
    return [center_x - math.sqrt(4 * c * f / (b ** 2 - 4 * a * c)),
            center_y - math.sqrt(4 * a * f / (b ** 2 - 4 * a * c)),
            center_x + math.sqrt(4 * c * f / (b ** 2 - 4 * a * c)),
            center_y + math.sqrt(4 * a * f / (b ** 2 - 4 * a * c))]


class FDDB(data.Dataset):
    image_folds = {
        'train': ['01', '02', '03', '04', '05'],
        'test': ['06', '07', '08', '09', '10'],
        'all': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    }
    cfg = {
        'num_classes': 2,
        'lr_steps': (2000, 4000, 6000, 8000),
        'max_iter': 200,
        'epochs': 20,
        'name': 'FDDB',
    }

    def __init__(self,
                 root=FDDB_ROOT,
                 image_list_txt='FDDB-fold-%s-ellipseList.txt',
                 dataset='train',
                 image_enhancement_fn=None
                 ):
        self.root = root
        self.image_list_txt = image_list_txt
        self.image_names = []
        self.image_rawBoxes = []
        self.num_of_each_fold = []
        self.image_enhancement_fn = image_enhancement_fn
        self.name = self.cfg['name']

        for fold in self.image_folds[dataset]:
            image_num_of_fold = 0
            f = open(path.join(root, image_list_txt % fold))
            line = f.readline()
            while line:
                image_num_of_fold += 1
                self.image_names.append(line.replace("\n", "") + '.jpg')
                line = f.readline()
                num_of_boxes = int(line)
                raw_boxes = []
                for i in range(num_of_boxes):
                    line = f.readline()
                    eclipse_para = line.split(' ')
                    raw_box = bounding_rect_of_eclipse(float(eclipse_para[0]),
                                                       float(eclipse_para[1]),
                                                       float(eclipse_para[2]),
                                                       float(eclipse_para[3]),
                                                       float(eclipse_para[4]))
                    raw_boxes.append(raw_box)
                self.image_rawBoxes.append(raw_boxes)
                line = f.readline()
            self.num_of_each_fold += [image_num_of_fold]

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
            image, boxes, classes = self.image_enhancement_fn(image, boxes_classes[:, :4], boxes_classes[:, 4])
            # to rgb
            image = image[:, :, (2, 1, 0)]
            boxes_classes = np.hstack((boxes, np.expand_dims(classes, axis=1)))
        return torch.from_numpy(image).permute(2, 0, 1), boxes_classes, h, w

    def pull_image(self, index):
        return cv2.imread(path.join(self.root, self.image_names[index]))

    def pull_anno(self, index):
        image_name = self.image_names[index]
        pass

    def pull_tensor(self, index):
        pass


if __name__ == '__main__':
    temp = FDDB(dataset='test')
    img, gt_boxes, height, width = temp.pull_item(1)
    print(gt_boxes)
