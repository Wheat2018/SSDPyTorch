from pycocotools.coco import COCO
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset 
import logging

logger = logging.getLogger(__name__)

class cocodataset(Dataset):

    def __init__(self, annotation_file, image_dir, cfg, transform=None, is_train=True, is_test=False):
        r"""
        annotation_file = '/path_to/coco/annotations/person_keypoints_train2017.json'
        image_dir = '/path_to/coco/images/train2017/' or '/path_to/coco/images/val2017/'
        """
        super(cocodataset,self).__init__()

        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
        self.input_size = (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT) # (w,h)
        self.random_scale = cfg.RANDOM_SCALE
        self.radius = cfg.RADIUS 
        self.kpts_num = cfg.NUM_KEYPOINTS
        self.edges = cfg.EDGES

        self.data = self.person_labeled_data()
        logger.info("===> total samples: {}".format(len(self.data)))
    
    def load_ann_into_memory(self,image_id):

        img_info = self.coco.loadImgs(image_id)[0]
        w = img_info['width']
        h = img_info['height']

        crowd_mask = np.zeros((h, w), dtype='bool')
        unannotated_mask = np.zeros((h,w), dtype='bool')

        instance_masks = []
        keypoints_skeletons = []
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        for ann in anns:
            if ann['area'] ==0  :
                continue
            mask = self.coco.annToMask(ann)

            if ann['iscrowd'] ==1:
                # paper:
                # we back-propagate across the full image, only excluding areas that
                # contain people that have not been fully annotated with keypoints (person crowd
                # areas and small scale person segments in the COCO dataset)
                crowd_mask = np.logical_or(crowd_mask, mask)

            elif ann['num_keypoints'] ==0:
                # paper:
                # Inputing missing keypoint annotations
                # The standard COCO dataset does not contain keypoint annotations for the small person instances
                # in the train set, and ignores them during model evaluation.

                # However, the small person has segmentation annotation and needs evaluateing mask predictions for instance segmentation,
                # So PersonLab uses G-RMI to predict missing keypoints for small instances
                # but here we ignore it
                unannotated_mask = np.logical_or(unannotated_mask, mask)
                
                instance_masks.append(mask)
                keypoints_skeletons.append(ann['keypoints'])
            else:
                instance_masks.append(mask)
                keypoints_skeletons.append(ann['keypoints'])
        
        return keypoints_skeletons, instance_masks, crowd_mask, unannotated_mask

    def person_labeled_data(self):
        data = []
        self.image_id_list = self.coco.getImgIds()  
        #print(self.image_id_list)
        ids_in_list = []
        for id, img_id in enumerate(self.image_id_list):
            file_name = self.coco.imgs[img_id]['file_name']
            file_path = os.path.join(self.image_dir, file_name)
            
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            for ann in anns:
                if ann['area'] == 0 or ann['num_keypoints'] == 0 : # change
                    continue
                
                sample = {

                    'img_id' : img_id,
                    'file_path' : file_path
                }

                # pick all images containing one person or more
                if img_id not in ids_in_list:
                    ids_in_list.append(img_id)
                    data.append(sample)

        return data

    def map_coco_to_personlab(self, keypoints):
        permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        if len(keypoints.shape) == 2:
            return keypoints[permute, :]
        return keypoints[:, permute, :]

    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, id):

        img_id  = self.data[id]['img_id']
        file_path =self.data[id]['file_path']
        
        keypoints, instance_masks, crowd_mask, unannotated_mask = self.load_ann_into_memory(img_id)    
        img = cv2.imread(file_path).astype('float32')
        h, w= img.shape[:2]
        instance_masks = np.array(instance_masks).transpose(1,2,0) # [num, H ,W ] TO [H, W, NUM]
        if self.is_train:
            scale = np.random.uniform(low=self.random_scale[0],high=self.random_scale[1])
            img_roi = [0, 0, w, h]
            affine_matrix = make_affine_matrix(img_roi,self.input_size,aug_scale=scale,aug_rotation=1)
            
            img = cv2.warpAffine(img, affine_matrix[0:2],self.input_size)
            keypoints = [kpt_affine(kpt, affine_matrix, self.input_size) for kpt in keypoints ]
            keypoints = np.array(keypoints) # [num,17,3]
            
            # pack masks into package to make warpaffine
            crowd_mask = np.expand_dims(crowd_mask,axis=-1)
            unannotated_mask = np.expand_dims(unannotated_mask,axis=-1)
            mask_package = np.concatenate([instance_masks, crowd_mask.astype('uint8'), unannotated_mask.astype('uint8')],axis=-1)
            # [H, W, Num] + [2,3] + [_W,_H]  --> [_H, _W, NUm]
            mask_package = cv2.warpAffine(mask_package,affine_matrix[0:2],self.input_size)
            
            instance_masks = mask_package[:,:,0:-2]
            crowd_mask = mask_package[:,:,-2].astype('bool')
            unannotated_mask = mask_package[:,:,-1].astype('bool')
        else:
            self.input_size = (w,h)
            keypoints = np.array(keypoints).reshape(-1,17,3)

        seg_mask = np.squeeze(np.sum(instance_masks,axis=-1)).astype('float32')

        instance_num = instance_masks.shape[-1]
        overlap_mask = np.zeros_like(seg_mask)

        if instance_num > 1:
            overlap_mask = instance_masks.sum(axis=-1) > 1
        
        unannotated_mask, crowd_mask, overlap_mask = \
            [np.logical_not(m).astype('float32') for m in [unannotated_mask, crowd_mask, overlap_mask]] 
        
        keypoints = self.map_coco_to_personlab(keypoints)
        #instance_masks = instance_masks.transpose(2,0,1) # [Num ,_H, _W]
        
        heatmaps, short_offsets, mid_offsets,  long_offsets = get_groundtruth(keypoints, 
                (self.input_size[1],self.input_size[0]), self.kpts_num, self.radius, self.edges, instance_masks)

        heatmaps = heatmaps.transpose(2,0,1)
        short_offsets = short_offsets.transpose(2,0,1)
        mid_offsets = mid_offsets.transpose(2,0,1)
        long_offsets = long_offsets.transpose(2,0,1)

        seg_mask = seg_mask[np.newaxis,:,:]
        crowd_mask = crowd_mask[np.newaxis,:,:]
        unannotated_mask =unannotated_mask[np.newaxis,:,:]
        overlap_mask = overlap_mask[np.newaxis,:,:]

        if self.transform:
            img = self.transform(img)
        #if self.is_train:
            #img = torch.from_numpy(img).cuda()
        return img, heatmaps, short_offsets, mid_offsets,long_offsets, \
            seg_mask,crowd_mask,unannotated_mask, overlap_mask, img_id



def make_affine_matrix(image_roi, target_size, margin=1.0, aug_rotation= 0, aug_scale=1):
    r"""
    transform image-roi to adapat the net-input size

    `margin`: determine the distance between the image and input border . default: 1.0
    `aug_rotation`: the rotation angle of augmentation, range from [0,180]. default: 0
    `aug_scale`: the rescale size of augmentation . default: 1

                            target_input_size
                             ____________
                            |   ______   |
                            |--|      |--|
                            |--|image |--|
                            |--|      |--|
                            |--|______|--|
                            |____________|

    t: 3x3 matrix, means transform image to input center-roi

    rs: 3x3 matrix , means augmentation of rotation and scale  

    Note that  

    """

    (w,h)=target_size

    #choose small-proportion side to make scaling
    scale = min((w/margin) /image_roi[2],
                (h/margin) /image_roi[3])
    
    #area = (w*h)/(scale*scale*margin*margin)

    # transform 
    t = np.zeros((3, 3))
    offset_X= w/2 - scale*(image_roi[0]+image_roi[2]/2)
    offset_Y= h/2 - scale*(image_roi[1]+image_roi[3]/2)
    t[0, 0] = scale
    t[1, 1] = scale
    t[0, 2] = offset_X
    t[1, 2] = offset_Y
    t[2, 2] = 1

    # augmentation
    theta = aug_rotation*np.pi/180
    alpha = np.cos(theta)*aug_scale
    beta = np.sin(theta)*aug_scale
    rs = np.zeros((3,3))
    rs[0, 0] = alpha
    rs[0, 1] = beta
    rs[0, 2] = (1-alpha)*(w/2)-beta*(h/2)
    rs[1, 0] = -beta
    rs[1, 1] = alpha
    rs[1, 2] = beta *(w/2) + (1-alpha)*(h/2)
    rs[2, 2] = 1
    
    # matrix multiply
    # first: t , orignal-transform
    # second: rs, augment scale and augment rotation
    final_matrix = np.dot(rs,t)
    return final_matrix  #, area


def kpt_affine(keypoints,affine_matrix,input_size):
    '[17*3] ==affine==> [17,3]'

    keypoints = np.array(keypoints).reshape(-1,3)
    for id,points in enumerate(keypoints):
        if points[2]==0:
            continue
        vis = points[2] # avoid python value bug
        points[2] = 1 # np.dot requires [x,y,1]
        keypoints[id][0:2] = np.dot(affine_matrix, points)[0:2]
        keypoints[id][2] = vis # np.dot requires homogeneous coordinates [x,y,1]

        if keypoints[id][0]<=0 or (keypoints[id][0]+1)>=input_size[0] or \
                keypoints[id][1]<=0 or (keypoints[id][1]+1)>=input_size[1]:
            keypoints[id][0]=0
            keypoints[id][1]=0
            keypoints[id][2]=0

    return keypoints


def symmetric_exchange_after_flipping(keypoints_flip, name):
    "flipping will make the left-right body parts exchange"

    if name == 'mpii':
        # for original mpii format 
        #parts = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        # for mpii to coco like
        parts = [[3,4],[5,6],[7,8],[9,10],[11,12],[13,14]]

    elif name == 'coco':
        parts = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]
    else:
        raise ValueError

    keypoints = keypoints_flip.copy()
    for part in parts:

        tmp = keypoints[part[1],:].copy()
        keypoints[part[1],:] = keypoints[part[0],:].copy()
        keypoints[part[0],:] = tmp

    return keypoints

def get_groundtruth(all_keypoints, map_shape, kpts_num, radius, edges, instance_masks= None):
    assert(instance_masks.shape[-1] == len(all_keypoints)), "{} != {}".format(
        instance_masks.shape[-1],len(all_keypoints))

    discs = get_keypoint_discs(all_keypoints, map_shape, kpts_num, radius, )

    kp_maps = make_keypoint_maps(all_keypoints,map_shape, discs,kpts_num)
    short_offsets = compute_short_offsets(all_keypoints,map_shape, discs,kpts_num, radius)
    mid_offsets = compute_mid_offsets(all_keypoints,map_shape, discs,edges)
    long_offsets = compute_long_offsets(all_keypoints,map_shape, instance_masks,kpts_num)

    return kp_maps, short_offsets, mid_offsets, long_offsets

def get_keypoint_discs(all_keypoints,map_shape,kpts_num,radius):
    #map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))

    discs = [[] for _ in range(len(all_keypoints))]
    for i in range(kpts_num):
        
        centers = [keypoints[i,:2] for keypoints in all_keypoints if keypoints[i,2] > 0]
        dists = np.zeros(map_shape+(len(centers),))

        for k, center in enumerate(centers):
            dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1))
        if len(centers) > 0:
            inst_id = dists.argmin(axis=-1)
        count = 0
        for j in range(len(all_keypoints)):
            if all_keypoints[j][i,2] > 0:
                discs[j].append(np.logical_and(inst_id==count, dists[:,:,count]<=radius))
                count +=1
            else:
                discs[j].append(np.array([]))

    return discs

def make_keypoint_maps(all_keypoints, map_shape, discs, kpts_num):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    kp_maps = np.zeros(map_shape+(kpts_num,))
    for i in range(kpts_num):
        for j in range(len(discs)):
            if all_keypoints[j][i,2] > 0:
                kp_maps[discs[j][i], i] = 1.
        
    return kp_maps