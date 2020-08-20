
import numpy as np
import cv2
from torch.utils import data
import scipy.io as sio


# https://github.com/JacobWang95/FaceAlignment_Base/blob/102e5f911cda5434b3274792069efc407800c03c/dataset.py

class AFLW(data.Dataset):
    def __init__(self, path_mat, data_root, size_img, channel='CHW', rgb_mean=(104, 117, 123), is_training=False):
        data = sio.loadmat(path_mat)
        self.name_list = data['nameList']
        self.kp = data['data']
        self.bbox = np.array(data['bbox'], int)
        self.ra = data['ra'] - 1
        self.size_img = size_img
        self.data_root = data_root
        self.rgb_mean = rgb_mean
        self.is_training = is_training
        self.channel = channel

    def __getitem__(self, index):
        if not self.is_training:
            index += 20000
        name_img = self.name_list[self.ra[0, index]]
        # print name_img
        # img = Image.open(self.data_root + name_img[0][0])
        # img = np.array(img)
        # if len(img.shape) == 2: img = np.tile(np.expand_dims(img, -1), (1,1,3))
        # img = img[:,:,[2,1,0]]
        img = cv2.imread(self.data_root + name_img[0][0])

        # print(name_img[0][0])

        img_h, img_w, img_c = img.shape
        size_b = np.array([img_w, img_w, img_h, img_h], np.float32)
        # bb [xmin, xmax, ymin, ymax]
        bb = self.bbox[self.ra[0, index]]
        # import pdb
        # pdb.set_trace()
        bb_w = bb[1]-bb[0]
        bb_h = bb[3]-bb[2]
        bb[0] = bb[0]-0.25*bb_w
        bb[1] = bb[1]+0.25*bb_w
        bb[2] = bb[2]-0.25*bb_h
        bb[3] = bb[3]+0.25*bb_h

        bb = np.maximum(bb, 0.0)
        bb = np.minimum(bb, size_b)
        bb = np.array(bb, int)
        # print bb
        img_crop = img[bb[2]:bb[3], bb[0]:bb[1], :]
        # print img_crop.shape

        img_crop = cv2.resize(img_crop, (self.size_img, self.size_img), interpolation=cv2.INTER_CUBIC)

        ori_size = bb[1] - bb[0]

        anno = self.kp[self.ra[0, index]]
        anno[0: 19] -= bb[0] #bb[0]表示xmin
        anno[19:] -= bb[2] #bb[2]表示ymin

        # anno[0: 19] /= (float(bb[1] - bb[0]) / float(self.size_img))
        # anno[19:] /= (float(bb[3] - bb[2]) / float(self.size_img))
        anno[0: 19] /= (float(bb[1] - bb[0]))
        anno[19:] /= (float(bb[3] - bb[2]))

        img_crop = img_crop.astype(np.float32)
        if self.rgb_mean:
            img_crop -= self.rgb_mean
        if self.channel == 'CHW':
            img_crop = np.transpose(img_crop, (2, 0, 1))
        return img_crop, anno, name_img[0][0]

    # 'AFLW': {'train': 20000, 'fullset': 24386, 'frontalset': 1314}
    def __len__(self):
        if self.is_training:
            return 20000
        else:
            return 4386


class AFLW_wo_trans(data.Dataset):
    def __init__(self, path_mat, data_root, is_training=False):
        data = sio.loadmat(path_mat)
        self.name_list = data['nameList']
        self.kp = data['data']
        self.bbox = np.array(data['bbox'], int)
        self.ra = data['ra'] - 1
        self.data_root = data_root
        self.is_training = is_training

    def __getitem__(self, index):
        if not self.is_training:
            index += 20000
        name_img = self.name_list[self.ra[0, index]]
        img = cv2.imread(self.data_root + name_img[0][0])
        img_h, img_w, img_c = img.shape
        size_b = np.array([img_w, img_w, img_h, img_h], np.float32)
        # bb [xmin, xmax, ymin, ymax]
        bb = self.bbox[self.ra[0, index]]
        ldmk = self.kp[self.ra[0, index]]
        return img, bb, ldmk, name_img[0][0]

    # 'AFLW': {'train': 20000, 'fullset': 24386, 'frontalset': 1314}
    def __len__(self):
        if self.is_training:
            return 20000
        else:
            return 4386








if __name__=="__main__":
    # train_data = AFLW('/home/gyt/dataset/AFLW/AFLWinfo_release.mat',
    #                           '/home/gyt/dataset/AFLW/data/flickr/', 256, is_training=True)
    #
    # test_data = AFLW('/home/gyt/dataset/AFLW/AFLWinfo_release.mat',
    #                          '/home/gyt/dataset/AFLW/data/flickr/', 256, is_training=False)

    # import pdb
    # pdb.set_trace()
    # data = sio.loadmat('/mnt/lustre/geyongtao/dataset/AFLW/AFLWinfo_release.mat')



    train_data = AFLW(path_mat='/mnt/lustre/geyongtao/dataset/AFLW/AFLWinfo_release.mat',
                      data_root='/mnt/lustre/geyongtao/dataset/AFLW/data/flickr/',
                      size_img=256,
                      is_training=True)

    test_data = AFLW(path_mat='/mnt/lustre/geyongtao/dataset/AFLW/AFLWinfo_release.mat',
                      data_root='/mnt/lustre/geyongtao/dataset/AFLW/data/flickr/',
                      size_img=256,
                      is_training=False)


    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
    for i, data in enumerate(train_loader):
        imgs, gt, img_name = data
        import pdb
        pdb.set_trace()



