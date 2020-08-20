import cv2
import sys
import os
import torch
import numpy as np
# from facedet.utils.misc import get_transform, kpt_affine
import torch.utils.data
from multiprocessing import dummy


class GenerateHeatmap():
    def __init__(self, output_res, num_parts=17, sigma_scale_for_invisible=2):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        self.sigma_scale_for_invisible = sigma_scale_for_invisible

    def gauss(self, hms, sigma, pt, idx):
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        x, y = int(pt[0]), int(pt[1])
        if x >= 0 and y >= 0 and x < self.output_res and y < self.output_res:
            ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
            br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

            c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res)

            hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b, c:d])

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma_scale_for_invisible = self.sigma_scale_for_invisible
        for p in keypoints:
            for idx, pt in enumerate(p):
                # import pdb
                # pdb.set_trace()
                if pt[2] == 2:  # 如果这个点不可见的话，是不是加入一个先验模型的大概率点？？
                    sigma = self.sigma
                    self.gauss(hms, sigma, pt, idx)

                if pt[2] == 1:  # 如果这个点不可见的话，是不是加入一个先验模型的大概率点？？

                    sigma = self.sigma * sigma_scale_for_invisible  # 对于不可见，但标注点，让heatmap高斯分布的方差变大，使得该点计算loss所占权重变小，或者该点容忍区域变大
                    self.gauss(hms, sigma, pt, idx)
        return hms

class KeypointsRef():
    def __init__(self, max_num_people, num_parts=17):
        self.max_num_people = max_num_people
        self.num_parts = num_parts

    def __call__(self, keypoints, output_res):
        visible_nodes = np.zeros((self.max_num_people, self.num_parts, 2))
        for i in range(len(keypoints)):
            tot = 0
            for idx, pt in enumerate(keypoints[i]):
                x, y = int(pt[0]), int(pt[1])
                if pt[2]>0 and x>=0 and y>=0 and x<output_res and y<output_res:
                    visible_nodes[i][tot] = (idx * output_res * output_res + y * output_res + x, 1)
                    tot += 1
        return visible_nodes

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opts, ds, index):
        self.input_res = opts.input_res 
        self.output_res = opts.output_res 

        self.generateHeatmap = GenerateHeatmap(opts.output_res)
        self.keypointsRef = KeypointsRef(opts.max_num_people)
        self.ds = ds
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.loadImage(self.index[idx % len(self.index)])

    def loadImage(self, idx):
        ds = self.ds

        inp = ds.load_image(idx)
        mask = ds.get_mask(idx).astype(np.float32)

        ann = ds.get_anns(idx)
        keypoints = ds.get_keypoints(idx, ann)

        keypoints = [i for i in keypoints if np.sum(i[:, 2]>0)>1]

        height, width = inp.shape[0:2]
        center = np.array((width/2, height/2))
        scale = max(height, width)/200

        inp_res = self.input_res
        res = (inp_res, inp_res)

        aug_rot = (np.random.random() * 2 - 1) * 30.
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale

        dx = np.random.randint(-40 * scale, 40 * scale)/center[0]
        dy = np.random.randint(-40 * scale, 40 * scale)/center[1]
        center[0] += dx * center[0]
        center[1] += dy * center[1]

        mat_mask = get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]
        mask = cv2.warpAffine((mask*255).astype(np.uint8), mat_mask, (self.output_res, self.output_res))/255
        mask = (mask > 0.5).astype(np.float32)

        mat = get_transform(center, scale, res, aug_rot)[:2]
        inp = cv2.warpAffine(inp, mat, res).astype(np.float32)/255
        keypoints[:,:,0:2] = kpt_affine(keypoints[:,:,0:2], mat_mask)

        if np.random.randint(2) == 0:
            inp = inp[:, ::-1]
            mask = mask[:, ::-1]
            keypoints = keypoints[:, ds.flipRef]
            keypoints[:, :, 0] = self.output_res - keypoints[:, :, 0]

        heatmaps = self.generateHeatmap(keypoints)
        keypoints = self.keypointsRef(keypoints, self.output_res)
        return self.preprocess(inp).astype(np.float32), mask.astype(np.float32), keypoints.astype(np.int32), heatmaps.astype(np.float32)

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        #cv2.imwrite('x.jpg', (data*255).astype(np.uint8))
        return data


def init(opts):
    batchsize = opts.batchsize
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_path)
    import ref as ds
    ds.init()

    train, valid = ds.setup_val_split()
    dataset = { key: Dataset(opts, ds, data) for key, data in zip( ['train', 'valid'], [train, valid] ) }

    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(dataset[key], batch_size=batchsize, shuffle=True, num_workers=opts.num_workers , pin_memory=False)

    def gen(phase):
        batchsize = opts.batchsize
        if phase=='train':
        	batchnum = opts.train_iters
        else:
        	batchnum = opts.valid_iters
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            imgs, masks, keypoints, heatmaps = next(loader)
            yield {
                'imgs': imgs,
                'masks': masks,
                'heatmaps': heatmaps,
                'keypoints': keypoints
            }


    return lambda key: gen(key)



if __name__=="__main__":
    keypoints =  [[[40.1, 50, 1],
                 [60.3, 86, 1],
                 [70.5, 20, 1],
                 [140, 120, 1],
                 [100, 150, 2]]]
    gen_heatmap = GenerateHeatmap(output_res=160, num_parts=5, sigma_scale_for_invisible=2)
    # heatmaps: chw
    heatmaps = gen_heatmap(keypoints)
    import pdb
    pdb.set_trace()
    for i in range(0, 5):
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmaps[i]), cv2.COLORMAP_JET)
        # import pdb
        # pdb.set_trace()
        cv2.imwrite('heatmap_'+str(i)+'.jpg', heatmap)
