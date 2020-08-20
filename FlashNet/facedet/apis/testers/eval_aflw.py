"""
python eval_aflw.py --trained_model ./weights/MobileFace_FPN_Smallest_Face_Landmark/Final_MobileFace.pth --cfg_file ./configs/mobileface_stu_fpn_smallest_face_landmark.py
"""
from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from facedet.utils.anchor.prior_box import PriorBox
from facedet.utils.ops.nms.nms_wrapper import nms
import cv2
from facedet.utils.bbox.box_utils import decode, decode_landmark, decode_ldmk
from facedet.utils.misc import Timer
from facedet.utils.bbox.fcos_target import FCOSTargetGenerator
# from facedet.utils.bbox.fcos_target import FCOSBoxConverter
from tensorboardX import SummaryWriter
import os
from dataset import AFLW
from torch.utils.data import DataLoader

# from mtcnn import detect_faces, show_bboxes
from facedet.utils.misc.checkpoint import *
from facedet.utils.eval.ldmk import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
writer = SummaryWriter('./test_log/')

target_generator = FCOSTargetGenerator()
parser = argparse.ArgumentParser(description='FaceBoxes')
parser.add_argument('--detector', default='ours', type=str, choices=['ours', 'mtcnn'], help='detector')
parser.add_argument('-m', '--trained_model', default='weights/MobileFace_FPN_Smallest_Face_Landmark/MobileFace_epoch_9.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cfg_file', default='./configs/mobileface_stu_fpn_smallest_face_5X5_landmark.py', type=str, help='model config file')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
parser.add_argument('--dataset', default='AFLW', type=str, choices=['AFLW'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=1, type=int, help='keep_top_k')
parser.add_argument('--path_mat', default='/mnt/lustre/geyongtao/dataset/AFLW/AFLWinfo_release.mat', type=str, help='aflw dataset mat path')
parser.add_argument('--data_root', default='/mnt/lustre/geyongtao/dataset/AFLW/data/flickr/', type=str, help='aflw dataset root path')
parser.add_argument('--size_img', default=256, type=int, help='image size')
args = parser.parse_args()


if __name__ == '__main__':
    # net and model
    from mmcv import Config
    cfg = Config.fromfile(args.cfg_file)
    save_folder = os.path.join('./eval/', args.dataset, cfg['test_cfg']['save_folder'])

    import models
    net = models.__dict__[cfg['net_cfg']['net_name']](phase='test', cfg=cfg['net_cfg'])
    net = load_model(net, args.trained_model)

    net.eval()
    print('Finished loading model!')
    print(net)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    # save file
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if args.dataset != 'WIDER':
        fw = open(os.path.join(save_folder, args.dataset + '_dets.txt'), 'w')

    # testing dataset
    if args.detector == 'mtcnn':
        test_data = AFLW(path_mat=args.path_mat, data_root=args.data_root,
                         size_img=args.size_img, channel='HWC', rgb_mean=None, is_training=False)
    else:
        test_data = AFLW(path_mat=args.path_mat, data_root=args.data_root,
                         size_img=args.size_img, is_training=False)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=4, shuffle=False)

    total_img = len(test_loader)
    num_valid_img = 0
    error_rate = []
    failure_count = 0
    max_threshold = 0.1

    _t = {'forward_pass': Timer(), 'misc': Timer()}


    # testing begin
    for i, data in enumerate(test_loader):


        iter_idx = i
        img, gt, img_name = data

        gt_landmarks = gt.reshape(2, 19).cpu().numpy().T*args.size_img
        gt_landmarks = np.stack((gt_landmarks[7], gt_landmarks[10], gt_landmarks[13], gt_landmarks[15], gt_landmarks[17]), axis=0)

        # (l_eye_center_index_x, l_eye_center_index_y)(14,15)
        # (r_eye_center_index_x, r_eye_center_index_y)(20,21)
        # (nose_center_index_x, nose_center_index_y)(26,27)
        # (mouth_left_corner_index_x, mouth_left_corner_index_x)(30,31)
        # (mouth_right_corner_index_x, mouth_right_corner_index_x)(38,39)

        img_h, img_w = img.size(2), img.size(3)
        origin_scale = torch.Tensor([img_w, img_h]).repeat(2)
        landmark_origin_scale = torch.Tensor([img_w, img_h]).repeat(5)

        if args.cuda:
            img = img.cuda()
            origin_scale = origin_scale.cuda()
            landmark_origin_scale = landmark_origin_scale.cuda()

        _t['forward_pass'].tic()
        # import pdb
        # pdb.set_trace()
        # landmark_delta, loc_delta, conf, feat_dim = net(img)  # forward pass
        landmark_delta, loc_delta, conf = net(img)  # forward pass
        _t['forward_pass'].toc()

        _t['misc'].tic()
        priorbox = PriorBox(cfg['anchor_cfg'], image_size=(img_h, img_w), phase='test')
        # priorbox = PriorBox(cfg['anchor_cfg'], feat_dim, (img_h, img_w), phase='test')
        priors = priorbox.forward()
        if args.cuda:
            priors = priors.cuda()
        # landmarks = decode_landmark(landmark_delta.squeeze(0), priors.data)
        landmarks = decode_ldmk(landmark_delta.squeeze(0), priors.data)
        boxes = decode(loc_delta.data.squeeze(0), priors.data, cfg['anchor_cfg']['variance'])
        boxes = boxes * origin_scale

        landmarks = landmarks * landmark_origin_scale
        landmarks = landmarks.data.cpu().numpy()

        boxes = boxes.data.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        if scores.max() < 0.1:
            continue
        num_valid_img = num_valid_img + 1
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]

        boxes = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        dets, nms_idx = nms(dets, args.nms_threshold)
        # keep top-K after NMS
        dets = dets[:args.keep_top_k, :]

        if cfg['net_cfg']['use_landmark']:
            landmarks = landmarks[inds]
            landmarks = landmarks[order]
            landmarks = landmarks[nms_idx]
            landmarks = landmarks[:args.keep_top_k, :]
        else:
            landmarks = None

        _t['misc'].toc()

        rgb_means = (104, 117, 123)
        img = img.squeeze(0).cpu().numpy().transpose(1, 2, 0) + rgb_means
        img = img.astype(np.uint8)
        img = img.copy()


        # for idx, point in enumerate(gt_landmarks):
        #     # print((int(point[0]), int(point[1])))
        #     cv2.circle(img, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=2)

        pred_landmarks = landmarks[0].reshape(5, 2)
        for idx, point in enumerate(pred_landmarks):
            # print((int(point[0]), int(point[1])))
            cv2.circle(img, (int(point[0]), int(point[1])), radius=3, color=(255, 0, 0), thickness=2)

        for box_score in dets:
            xmin, ymin, xmax, ymax, score = box_score
            if score > 0.5:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img, str('%0.2f' % score), (xmin, ymax), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0),1)

        img_write_path = os.path.join('./eval', 'write_aflw', img_name[0].split('/')[0])
        if not os.path.exists(img_write_path):
            os.makedirs(img_write_path)
        img_name = os.path.join(img_write_path,img_name[0].split('/')[1])
        cv2.imwrite(img_name, img)

        error_normalize_factor = np.sqrt((xmax-xmin)*(ymax-ymin))
        # import pdb
        # pdb.set_trace()
        error_rate_i = calc_error_rate_i(pred_landmarks, gt_landmarks, error_normalize_factor)
        failure_count = failure_count + 1 if error_rate_i > max_threshold else failure_count
        error_rate.append(error_rate_i)
        if i % 1 == 0:
            print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, total_img,
                                                                                         _t['forward_pass'].average_time,
                                                                                         _t['misc'].average_time))

    area_under_curve, auc_record = calc_auc(num_valid_img, error_rate, max_threshold)
    error_rate = sum(error_rate) / num_valid_img * 100
    failure_rate = failure_count / num_valid_img * 100
    eval_CED(auc_record)
    # print(
    #       'Error Rate: ' + str(error_rate) + '%\n' +
    #       'Failure Rate: ' + str(failure_rate) + '%\n')

    print('AUC: ' + str(area_under_curve) + '\n' +
          'Error Rate: ' + str(error_rate) + '%\n' +
          'Failure Rate: ' + str(failure_rate) + '%\n')