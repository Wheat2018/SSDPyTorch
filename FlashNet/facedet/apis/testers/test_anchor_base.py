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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
writer = SummaryWriter('./test_log/')

target_generator = FCOSTargetGenerator()
parser = argparse.ArgumentParser(description='FaceBoxes')

parser.add_argument('-m', '--trained_model', default='../../../../weights/FlashNet_WIDER.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cfg_file', default='../../configs/flashnet_1024_2_anchor.py', type=str, help='model config file')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
parser.add_argument('--dataset', default='WIDER', type=str, choices=['AFW', 'PASCAL', 'FDDB', 'WIDER',
                                                                               'WIDER_train_5','WIDER_test'], help='dataset')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def save_roc_file(image_path, bboxes_scores, output_dir='./eval/preds_stu'):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        image_path (string): file name.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]

    odir = output_dir
    if not os.path.exists(os.path.join(odir, image_class)):
        os.makedirs(os.path.join(odir, image_class))

    ofname = os.path.join(odir, img_name + '.txt')
    f = open(ofname, 'w')
    #     f.write('{:s}\n'.format(image_class + '/' + image_name))
    #     f.write('{:d}\n'.format(bboxes_scores.shape[0]))
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        f.write('{:s} {:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(image_name, xmin, ymin, xmax, ymax, score))
    f.close()


def save_widerface_bboxes(image_path, bboxes_scores, output_dir, iteration, landmark_pos=None, draw_landmark=None):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        image_path (string): file name.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    image_name = image_path.split('/')[-1]
    image_class = image_path.split('/')[-2]

    odir = os.path.join(output_dir, image_class)
    if not os.path.exists(odir):
        os.makedirs(odir)

    ofname = os.path.join(odir, '%s.txt' % (image_name[:-4]))
    f = open(ofname, 'w')
    f.write('{:s}\n'.format(image_class + '/' + image_name))
    f.write('{:d}\n'.format(bboxes_scores.shape[0]))

    img_origin = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
    #     print(image_path)
    # import pdb
    # pdb.set_trace()
    # lanp.concatenate(landmarks,bboxes_scores[:,-1], axis=1)

    if draw_landmark:
        for idx, landmark in enumerate(landmark_pos):
            landmark = landmark.reshape(5,2)
            score = bboxes_scores[idx, -1]
            if score > 0.4:
                for point in landmark:
                    # import pdb
                    # pdb.set_trace()
                    # print('point',point)
                    cv2.circle(img_origin,(int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=2)

    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax, score = box_score
        f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(xmin, ymin, (
                xmax - xmin + 1), (ymax - ymin + 1), score))
        if score > 0.4:
            if xmax - xmin < 25 and ymax - ymin < 25:
                cv2.rectangle(img_origin, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img_origin, str('%0.2f' % score), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),
                            1)
            elif xmax - xmin < 35 and ymax - ymin < 35:
                cv2.rectangle(img_origin, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img_origin, str('%0.2f' % score), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 255), 1)
            else:
                cv2.rectangle(img_origin, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img_origin, str('%0.2f' % score), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    # img_write_path = os.path.join('./eval', 'write_img_with_bbox_landmark', image_class)
    img_write_path = os.path.join('./eval', 'write_img_with_bbox', image_class)
    if not os.path.exists(img_write_path):
        os.makedirs(img_write_path)
    img_write_name = os.path.join(img_write_path, image_name)
    cv2.imwrite(img_write_name, img_origin)

    img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB) / 255
    # writer.add_image('img_with_pred_box', img_origin,
    #                  global_step=iteration, dataformats='HWC')

    f.close()



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

    if 'WIDER' in args.dataset:
        testset_folder = os.path.join('../../../../data', args.dataset, 'JPEGImages/')
        testset_list = os.path.join('../../../../data', args.dataset, 'ImageSets/Main/test.txt')
    elif args.dataset == 'FDDB' or args.dataset == 'AFW' or args.dataset == 'PASCAL':
        testset_folder = os.path.join('../../../../data', args.dataset, 'images/')
        testset_list = os.path.join('../../../../data', args.dataset, 'img_list.txt')


    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    # testing scale
    if args.dataset == "FDDB":
        resize = 3
    elif args.dataset == "PASCAL":
        resize = 2.5
    elif args.dataset == "AFW" or "WIDER" in args.dataset:
        resize = 1

    _t = {'forward_pass': Timer(), 'misc': Timer()}

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
        if "WIDER" in args.dataset:
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

        _t['forward_pass'].tic()
        with torch.no_grad():
            out = net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()

        if cfg['test_cfg']['is_anchor_base'] is True:
            priorbox = PriorBox(cfg['anchor_cfg'], image_size=(im_height, im_width), phase = 'test')
            priors = priorbox.forward()
            if args.cuda:
                priors = priors.cuda()

            if cfg['train_cfg']['use_landmark']:
                loc, conf, landmark_delta = out
                # landmarks = decode_landmark(landmark_delta.squeeze(0), priors.data)
                landmarks = decode_ldmk(landmark_delta.squeeze(0), priors.data)

            else:
                # loc, conf, _ = out
                loc, conf = out

            boxes = decode(loc.data.squeeze(0), priors.data, cfg['anchor_cfg']['variance'])
            # pdb.set_trace()
            # if cfg['test_cfg']['is_ctr'] is True:
            #     boxes = boxes[prior_mask.repeat(1,4)].view(-1,4)
            #     conf = conf[prior_mask.repeat(1, 2)].view(-1, 2)
        else:
            box_preds, conf, centerness, _ = out

            #             import pdb
            #             pdb.set_trace()
            #             conf = conf*centerness
            box_cords = target_generator.generate_box_cords(im_height, im_width)
            if args.cuda:
                box_cords = box_cords.cuda()
            #             import pdb
            #             pdb.set_trace()
            #             box_preds = box_preds * origin_scale
            #             boxes = FCOSBoxConverter(box_preds.exp(), box_cords.unsqueeze(0)).squeeze(0)
            #             import pdb
            #             pdb.set_trace()
            boxes = FCOSBoxConverter(box_preds, box_cords.unsqueeze(0), im_height, im_width).squeeze(0)

        if args.dataset == 'FDDB' or args.dataset == 'PASCAL' or args.dataset == 'AFW':
            # boxes = boxes * scale / resize
            boxes = boxes * origin_scale
            boxes = boxes.detach().cpu().numpy()
        else:
            if cfg['test_cfg']['is_anchor_base'] is True:
                boxes = boxes * origin_scale
                if cfg['train_cfg']['use_landmark']:
                    landmarks = landmarks * landmark_origin_scale
                    landmarks = landmarks.data.cpu().numpy()
            #                 boxes = boxes * origin_scale / resize
            else:
                boxes = boxes / resize
            boxes = boxes.data.cpu().numpy()

        conf = conf.data.squeeze(0)
        #         import pdb
        #         pdb.set_trace()
        if cfg['net_cfg']['num_classes'] == 1 and cfg['test_cfg']['is_anchor_base'] == False:
            #             import pdb
            #             pdb.set_trace()
            scores = conf.data.cpu().numpy()[:, 0]
        else:
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
        #         keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
        #         keep = nms(dets, args.nms_threshold)
        #         dets = dets[keep, :]
        dets, nms_idx = nms(dets, args.nms_threshold)
        # keep top-K after NMS
        dets = dets[:args.keep_top_k, :]

        if cfg['train_cfg']['use_landmark']:
            landmarks = landmarks[inds]
            landmarks = landmarks[order]
            landmarks = landmarks[nms_idx]
            landmarks = landmarks[:args.keep_top_k, :]
        else:
            landmarks = None

        _t['misc'].toc()

        # save dets
        if "WIDER" in args.dataset:
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

            continue
            if cfg['train_cfg']['use_landmark']:
                draw_landmark = True
            else:
                draw_landmark = False

            save_widerface_bboxes(image_path, dets, save_folder, iter_idx, landmarks, draw_landmark=draw_landmark)

        #             save_roc_file(image_path, dets)
        img_origin = np.float32(cv2.imread(image_path, cv2.IMREAD_COLOR))
        if args.dataset == "FDDB":
            fw.write('{:s}\n'.format(img_name))
            fw.write('{:.1f}\n'.format(dets.shape[0]))
            for k in range(dets.shape[0]):
                xmin = dets[k, 0]
                ymin = dets[k, 1]
                xmax = dets[k, 2]
                ymax = dets[k, 3]
                score = dets[k, 4]
                w = xmax - xmin + 1
                h = ymax - ymin + 1
                fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
                if score > 0.5:
                    cv2.rectangle(img_origin, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                    cv2.putText(img_origin, str('%0.2f' % score), (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            img_write_path = os.path.join('./eval', 'fddb')
            if not os.path.exists(img_write_path):
                os.makedirs(img_write_path)
            image_name = image_path.split('/')[-1]
            img_write_name = os.path.join(img_write_path, image_name)
            cv2.imwrite(img_write_name, img_origin)



        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images,
                                                                                     _t['forward_pass'].average_time,
                                                                                     _t['misc'].average_time))


