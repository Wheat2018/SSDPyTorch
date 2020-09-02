"""
python test_wider.py -m ./weights/ --cfg_file
python test_debug.py --cfg_file ./configs/shufflenetv2_fpn_enhance_1024.py --trained_model ./weights/ShuffleNetv2FPN1024/epoch_196.pth --dataset FDDB
"""
from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
from FlashNet.facedet.utils.ops.nms.nms_wrapper import nms
import cv2
from FlashNet.facedet.utils.bbox.box_utils import decode, decode_landmark, decode_ldmk
from FlashNet.facedet.utils.misc import Timer
from FlashNet.facedet.utils.bbox.fcos_target_old import FCOSTargetGenerator
from FlashNet.facedet.utils.bbox.fcos_target_old import FCOSBoxConverter
from tensorboardX import SummaryWriter
import os
from FlashNet.facedet.utils.misc.checkpoint import *
from layers.box_utils import nms as nms2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
writer = SummaryWriter('./test_log/')

target_generator = FCOSTargetGenerator()
parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='weights/FlashNet_WIDER.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cfg_file', default='FlashNet/facedet/configs/flashnet_1024_2_anchor.py', type=str, help='model config file')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
parser.add_argument('--dataset', default='WIDER', type=str, choices=['AFW', 'PASCAL', 'FDDB', 'WIDER',
                                                                               'WIDER_train_5','WIDER_test'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()

def detect(out, priors, variance, scale, top_k=200, conf_thresh=0.1, nms_thresh=0.45):

    loc_data, conf_data = out
    """
        1: loc_data, Shape: [batch_num,priors_num,4]
        2: conf_data, Shape: [batch_num,priors_num, classes_num]
        3: priors_data, Shape: [priors_num,4]
    """
    batch_num = conf_data.shape[0]
    priors_num = conf_data.shape[1]
    classes_num = conf_data.shape[2]
    if top_k is None or top_k <= 0:
        top_k = priors_num
    output = []

    # Decode predictions into bboxes.
    for i in range(batch_num):
        decoded_boxes = decode(loc_data[i], priors, variance)
        # decoded_boxes *= scale
        # For each class, perform nms
        conf_scores = conf_data[i].clone().t()      # [classes_num, priors_num]

        output_each = torch.Tensor()
        if args.cuda:
            output_each = output_each.cuda()
        for cl in range(1, classes_num):
            conf_of_cl = conf_scores[cl]
            c_mask = conf_of_cl.gt(conf_thresh)
            scores = conf_of_cl[c_mask]
            if scores.size(0) == 0:
                output += [torch.Tensor()]
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4)
            # idx of highest scoring and non-overlapping boxes per class
            ids, count = nms2(boxes, scores, nms_thresh, top_k)
            output_cl = torch.cat((boxes[ids[:count]],
                                 scores[ids[:count]].unsqueeze(1)), 1)
            if classes_num > 2:
                output_cl = torch.cat((torch.Tensor([cl]).expand(count, 1),
                                       output_cl), 1)
            output_each = torch.cat((output_each, output_cl), 0)
        output += [output_each]
    return output

if __name__ == '__main__':
    # net and model
    from mmcv import Config
    cfg = Config.fromfile(args.cfg_file)
    save_folder = os.path.join('./eval/', args.dataset, cfg['test_cfg']['save_folder'])
    import FlashNet.facedet.models as models
    net = models.__dict__[cfg['net_cfg']['net_name']](phase='test', cfg=cfg['net_cfg'])
    net = load_model(net, args.trained_model)

    net.eval()
    print('Finished loading model!')
    print(net)

    # if args.tvm_time_benchmark:
    #     tvm_forward_time_test(net)

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

    testset_folder = os.path.join('data', args.dataset, 'JPEGImages/')
    testset_list = os.path.join('data', args.dataset, 'ImageSets/Main/test.txt')


    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # name_list = ["0_Parade_Parade_0_519.jpg"]

    # testing begin
    for i, img_name in enumerate(test_dataset):
        iter_idx = i
        image_path = testset_folder + img_name + '.jpg'
        #         print(image_path)
        # import pdb
        # pdb.set_trace()
        img = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
        #         img_origin = img
        origin_scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        landmark_origin_scale = torch.Tensor([img.shape[1], img.shape[0]]).repeat(5)

        # import pdb
        # pdb.set_trace()
        resize = 1600 / img.shape[0]
        # resize = 1080 / img.shape[0]
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        if args.cuda:
            img = img.cuda()
            scale = scale.cuda()
            origin_scale = origin_scale.cuda()
            landmark_origin_scale = landmark_origin_scale.cuda()

        with torch.no_grad():
            out = net(img)  # forward pass

        priorbox = PriorBox(cfg['anchor_cfg'], image_size=(im_height, im_width), phase = 'test')
        priors = priorbox.forward()
        if args.cuda:
            priors = priors.cuda()

        # dets = detect(out, priors.data, cfg['anchor_cfg']['variance'], origin_scale)
        # dets = dets[0]
        # dets[:, :-1] *= origin_scale
        # dets = dets.data.cpu().numpy()
        loc, conf = out

        boxes = decode(loc.data.squeeze(0), priors.data, cfg['anchor_cfg']['variance'])

        boxes = boxes * origin_scale
        boxes = boxes.data.cpu().numpy()

        conf = conf.data.squeeze(0)
        #         import pdb
        #         pdb.set_trace()
        scores = conf.data.cpu().numpy()[:, 1]

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

        # save dets
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        color = [0, 0, 255]
        for n in range(dets.shape[0]):
            score = dets[n, -1]
            if score < 0.4:
                continue
            display_txt = '%.2f' % score
            pt = (torch.Tensor(dets[n, :-1])).type(torch.int32).cpu().numpy()

            cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
            # cv2.fillPoly(image,
            #              np.array([[[pt[0], pt[1]], [pt[0] + 25, pt[1]], [pt[0] + 25, pt[1] + 15],
            #                         [pt[0], pt[1] + 15]]]),
            #              color)
            # inverse_color = [255 - x for x in color]
            # cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, inverse_color, lineType=cv2.LINE_AA)
        cv2.imshow('test', image)
        k = cv2.waitKey(0)
        if k == 27:
            break

