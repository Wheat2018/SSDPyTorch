"""
    By.Wheat
    2020.05.11
"""

import torch.utils.data as data
from common import *

FDDB_ROOT = path.join(DATA_ROOT, 'FDDB')


def bounding_rect_of_ellipse(major_axis_radius,
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
        'max_iter': 12000,
        'epochs': 20,
    }

    def __init__(self,
                 root=FDDB_ROOT,
                 image_list_txt='FDDB-fold-%s-ellipseList.txt',
                 dataset='train',
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
            f = open(path.join(root, image_list_txt % fold))
            line = f.readline()
            while line:
                self.image_names.append(line.replace("\n", "") + '.jpg')
                line = f.readline()
                num_of_boxes = int(line)
                raw_boxes = []
                for i in range(num_of_boxes):
                    line = f.readline()
                    ellipse_para = line.split(' ')
                    raw_box = bounding_rect_of_ellipse(float(ellipse_para[0]),
                                                       float(ellipse_para[1]),
                                                       float(ellipse_para[2]),
                                                       float(ellipse_para[3]),
                                                       float(ellipse_para[4]))
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

    def write_eval_result(self, save_folder, save_filename='fold-%s-out.txt', fold=None):
        result = []
        if fold is None:
            fold = self.image_folds[self.dataset]

        if isinstance(fold, list):
            for f in fold:
                result += self.write_eval_result(save_folder, save_filename, f)
        else:
            filename = path.join(save_folder, save_filename % fold)
            with open(filename, 'wt') as file:
                idx_range = self.range_of_each_fold[fold]
                for idx in range(idx_range[0], idx_range[1]):
                    image_name = self.image_names[idx]
                    image_name, _ = os.path.splitext(image_name)

                    file.write(image_name + '\n')
                    file.write(str(len(self.image_preBoxes[idx])) + '\n')
                    for box in self.image_preBoxes[idx]:
                        line = ''
                        for e in box:
                            line += '%.6f ' % e
                        file.write(line + '\n')
            result.append(filename)
        return result

    def sign_item(self, index, boxes_conf, h, w):
        """
        :param index: image index
        :param boxes_conf: tensor[boxes_num, 5]
                           for each boxes_conf: [conf, left, top, right, bottom]
        :param h: int. image original height
        :param w: int. image original width
        :return: void
        """
        boxes = []
        scale = torch.Tensor([w, h, w, h])
        for i in range(boxes_conf.shape[0]):
            box = torch.Tensor(boxes_conf[i, 1:]) * scale
            box = torch.Tensor([box[0], box[1],
                                box[2] - box[0],
                                box[3] - box[1],
                                boxes_conf[i, 0]])
            boxes.append(box)
        self.image_preBoxes[index] = boxes

    def pull_item(self, index):
        """
        :param index: image index
        :return: (image: tensor[3, 300, 300], boxes_classes: list[boxes_num, 5],
                  image original height: int, image original width: int)
                 for each boxes_classes: [left, top, right, bottom, 0]
        """
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
