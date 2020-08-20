'''
run command:

python ./tools/roc.py --anno_file=/home/gyt/vehicle/ImageSets/Main/test.txt --save_dir=./eval --anno_key=vehicle --ious=0.5 --scale_ranges="(0.0,1.0)" --label_info

'''
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import sys
import argparse

sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
from xml.dom import minidom

def parse_args():
    parser = argparse.ArgumentParser(description='roc')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='/home/gyt/dataset/vehicle', help='dataset root dir', type=str)
    parser.add_argument('--anno_file', dest='anno_file', help='gt file path', type=str)
    parser.add_argument('--anno_key', dest='anno_key', help='object name in xml, use it to get gt', type=str)
    parser.add_argument('--save_dir', dest='save_dir', default='./eval', help='save dir', type=str)
    parser.add_argument('--ious', dest='ious', default="(0.3,0.4)", help='ious to compute roc, should be tuple or list',
                        type=str)
    parser.add_argument('--scale_ranges', dest='scale_ranges', default="(0.0,1.0)",
                        help='split range scales to compute roc, "(s1,s2),(s3,s4),..."', type=str)
    parser.add_argument('--label_info', dest='label_info',
                        help='flag to note whether have label information in pred_file', action='store_true')
    parser.add_argument('--show', dest='show', help='whether to show plot figure', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args


prob_thres = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.905, 0.910,
              0.915, 0.920, 0.925, 0.930, 0.935, 0.940, 0.945, 0.950, 0.955, 0.960,
              0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 0.999]


def iou(b1, b2):
    iou_val = 0.0
    x1 = np.max([b1[0], b2[0]])
    y1 = np.max([b1[1], b2[1]])
    x2 = np.min([b1[0] + b1[2], b2[0] + b2[2]])
    y2 = np.min([b1[1] + b1[3], b2[1] + b2[3]])
    #     x2 = np.min([b1[2], b2[2]])
    #     y2 = np.min([b1[3], b2[3]])
    w = np.max([0, x2 - x1])
    h = np.max([0, y2 - y1])

    if w != 0 and h != 0:
        iou_val = float(w * h) / (b1[2] * b1[3] + b2[2] * b2[3] - w * h)
    #         print(iou_val)

    return iou_val


def precision_recall(pred, gt, thres):
    pn = len(pred)
    rn = len(gt)
    #     print('len(gt)',rn)
    pm = 0
    rm = 0
    pred[:, 2] = pred[:, 2] - pred[:, 0]
    pred[:, 3] = pred[:, 3] - pred[:, 1]
    #     gt[:,2] = gt[:,0]+gt[:,2]
    #     gt[:,3] = gt[:,1]+gt[:,3]
    for b1 in pred:
        for b2 in gt:
            #             print('pred',b1)
            #             print('gt',b2)
            #             import pdb
            #             pdb.set_trace()
            #             print('iou(b1, b2)',iou(b1, b2))
            if iou(b1, b2) > thres:
                pm += 1
                break
    for b1 in gt:
        for b2 in pred:
            if iou(b1, b2) > thres:
                rm += 1
                break

    return pm, rm, pn, rn


def get_gt(file_path, anno_key, scale_range_size):
    gt = np.empty((0, 4), dtype=np.int32)
    annotation = minidom.parse(file_path).documentElement
    objectlist = annotation.getElementsByTagName('object')
    for obj in objectlist:
        namelist = obj.getElementsByTagName('name')
        objname = namelist[0].childNodes[0].data
        if objname == anno_key or objname == 'Vehicle':
            bndbox = obj.getElementsByTagName('bndbox')
            for box in bndbox:
                xminlist = box.getElementsByTagName('xmin')
                xmin = int(float(xminlist[0].childNodes[0].data))
                yminlist = box.getElementsByTagName('ymin')
                ymin = int(float(yminlist[0].childNodes[0].data))
                xmaxlist = box.getElementsByTagName('xmax')
                xmax = int(float(xmaxlist[0].childNodes[0].data))
                ymaxlist = box.getElementsByTagName('ymax')
                ymax = int(float(ymaxlist[0].childNodes[0].data))
                w = xmax - xmin + 1
                h = ymax - ymin + 1

                gt_box = np.array([[xmin, ymin, w, h]], dtype=np.int32)
                gt = np.concatenate((gt, gt_box), axis=0)
    inds = np.where((gt[:, 3] >= scale_range_size[0]) & (gt[:, 3] <= scale_range_size[1]))[0]
    gt = gt[inds, :]

    return gt


def get_pred_txt(file_path, label_info, scale_range_size):
    probs = []
    bbs = []

    with open(file_path, 'r+') as f:

        try:
            for line in f.readlines():
                # line = f.next().strip()
                lst = line.split()
                if label_info:
                    pred = list(map(float, lst[1:5]))
                    score = float(lst[5])
                else:
                    pred = list(map(int, lst[:4]))
                    score = float(lst[4])
                bbs.append(pred)
                probs.append(score)
            '''
            while True:
                line = f.next().strip()
                lst = line.split()
                if label_info:
                    pred = map(float, lst[1:5])
                    score = float(lst[5])
                else:
                    pred = map(int, lst[:4])
                    score = float(lst[4])
                bbs.append(pred)
                probs.append(score)
            '''
        except StopIteration:
            pass

    bbs = np.array(bbs)

    # bbs[:,2] = bbs[:,2] - bbs[:,0]
    # bbs[:,3] = bbs[:,3] - bbs[:,1]

    probs = np.array(probs)
    if len(bbs) > 0:
        inds = np.where((bbs[:, 3] >= scale_range_size[0]) & (bbs[:, 3] <= scale_range_size[1]))[0]
        bbs = bbs[inds, :]
        probs = probs[inds]

    return bbs, probs


def show(img_id, gt, pred, prob, thres=0.6):
    img_path = os.path.join(args.dataset_dir, 'JPEGImages/' + img_id + ".jpg")
    img = cv2.imread(img_path)
    #     import pdb
    #     pdb.set_trace()
    #     if img_id == '2253':
    #         pdb.set_trace()
    for bbox in gt:
        # xmin, ymin, w, h
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
    #         cv2.imshow("gt", img)

    if len(pred) > 0 and len(prob) > 0:
        I = prob > thres
        prob_k = prob[I]
        pred_k = pred[I, :]
        for bbox, pb in zip(pred_k, prob_k):
            bbox = bbox.astype(np.int)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(img, str(pb), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    img_name = './eval/write_gt_pred/' + img_id + '.jpg'
    cv2.imwrite(img_name, img)


#     key = cv2.waitKey(0)
#     if key == 27:
#         sys.exit(0)


def roc(gt_pred_file, save_dir, anno_key, iou_thres, label_info, scale_range):
    lines = open(gt_pred_file, 'r').readlines()
    precisions = np.zeros((len(prob_thres),), dtype=np.int32)
    recalls = np.zeros((len(prob_thres),), dtype=np.int32)
    phits = np.zeros((len(prob_thres),), dtype=np.int32)
    rhits = np.zeros((len(prob_thres),), dtype=np.int32)
    cnt = 0
    for line in lines:
        img_id, pred_file = line.split()
        gt_file = os.path.join(args.dataset_root, 'Annotations/' + img_id + '.xml')
        #         import pdb
        #         pdb.set_trace()
        annotation = minidom.parse(gt_file).documentElement
        sizelist = annotation.getElementsByTagName('size')
        for size in sizelist:
            widthlist = size.getElementsByTagName('width')
            width = widthlist[0].childNodes[0].data
            heightlist = size.getElementsByTagName('height')
            height = heightlist[0].childNodes[0].data
            depthlist = size.getElementsByTagName('depth')
            depth = depthlist[0].childNodes[0].data
        scale_range_size = float(height) * np.array(scale_range, dtype=np.float32)  # 将在scale范围内的gt提取出来
        gt = get_gt(gt_file, anno_key, scale_range_size)
        pred, probs = get_pred_txt(pred_file, label_info, scale_range_size)  # 将在scale范围内的pred提取出来


        for k in range(len(prob_thres)):
            if len(pred) > 0:
                I = probs > prob_thres[k]
                prob_k = probs[I]
                pred_k = pred[I, :]
            else:
                pred_k = []
            pm, rm, pn, rn = precision_recall(pred_k, gt, iou_thres)
            phits[k] += pm
            rhits[k] += rm
            precisions[k] += pn
            recalls[k] += rn
        #         print('phits',phits)
        #         print('rhits',rhits)
        #         print('precisions',precisions)
        #         print('recalls',recalls)

        #         show(img_path, img_id,  gt, pred, probs)

        cnt += 1
        if cnt % 1 == 0:
            print("precessed {}".format(cnt))

    roc_save_dir = "{}/roc".format(save_dir)
    if os.path.exists(roc_save_dir):
        shutil.rmtree(roc_save_dir)
    os.mkdir(roc_save_dir)

    mean_p = np.zeros((len(prob_thres),), dtype=np.float32)
    mean_r = np.zeros((len(prob_thres),), dtype=np.float32)
    for k in range(len(prob_thres)):
        if precisions[k] == 0:
            mean_p[k] = 1.0
            continue
        if recalls[k] == 0:
            mean_p[k] = 1.0 / (precisions[k] + 1)
            continue

        if recalls[k] == 0:
            mean_r[k] = 1.0
            continue
        if precisions[k] == 0:
            mean_r[k] = 0.0
            continue

        mean_p[k] = phits[k] * 1.0 / precisions[k]
        mean_r[k] = rhits[k] * 1.0 / recalls[k]

    np.save("%s/precision_iou%.3f.npy" % (roc_save_dir, iou_thres), mean_p)
    np.save("%s/recall_iou%.3f.npy" % (roc_save_dir, iou_thres), mean_r)

    return mean_p, mean_r


def get_gt_pred_file(gt_path, level, pred_dir, save_dir):
    lines = open(gt_path, 'r').readlines()
    gt_pred_file = "{}/gt_pred_file".format(save_dir)
    #     import pdb
    #     pdb.set_trace()
    f = open(gt_pred_file, 'w')
    for line in lines:
        print(line)
        lst = line.split()
        if len(lst) == 1:
            line = line.strip()
        else:
            line = line.split()[1].strip()
        inx = min(len(line.split('/')), level)
        anno_name = '/'.join(line.split('/')[-inx - 1:]).split('.')[0] + '.txt'
        f.write("{} {}/{}\n".format(line, pred_dir, anno_name))
    f.close()

    return gt_pred_file





def get_title(save_dir):
    lst = os.path.abspath(save_dir).split('/')
    index = lst.index('nemo_eval')
    title_str = '/'.join(lst[index - 1:])

    return title_str


if __name__ == "__main__":
    args = parse_args()
    pred_dir = "{}/preds".format(args.save_dir)
    gt_pred_file = get_gt_pred_file(args.anno_file, 1, pred_dir, args.save_dir)
    scale_range_list = eval(args.scale_ranges)
    if not isinstance(scale_range_list[0], list) and not isinstance(scale_range_list[0], tuple):
        scale_range_list = (scale_range_list,)

    iou_list = eval(args.ious)
    if not isinstance(iou_list, list) and not isinstance(iou_list, tuple):
        iou_list = (iou_list,)

    plt.figure(figsize=(8, 7))
    stats = []
    for iou_thres in iou_list:
        for scale_range in scale_range_list:
            mean_p, mean_r = roc(gt_pred_file, args.save_dir, args.anno_key, iou_thres, args.label_info, scale_range)
            stats.append([mean_p, mean_r])
    for mean_p, mean_r in stats:
        plt.plot(mean_p, mean_r, 'o-')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.title('Vehicle detection model ROC curve')
    # plt.title(get_title(args.save_dir))
    plt.legend(['iou: {} scale: [{}, {}]'.format(iou_thres, scale_range[0], scale_range[1])
                for iou_thres in iou_list for scale_range in scale_range_list], loc="lower left")
    plt.savefig("{}/roc/roc.png".format(args.save_dir))
    if args.show:
        plt.show()
