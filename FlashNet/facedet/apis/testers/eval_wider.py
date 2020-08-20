from __future__ import print_function
from scipy import io as sio
import numpy as np
import copy
import multiprocessing
import logging
import argparse
import pdb
from functools import reduce
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='WIDER EVAL')

parser.add_argument('-d', '--det_dir', default='./eval', type=str, help='WIDER detections output directory')
parser.add_argument('-g', '--gt_dir', default='/mnt/lustre/geyongtao/dataset/WIDER_train_5', type=str,
                    help='WIDER ground truth output directory')
parser.add_argument('--save_dir', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='iou_threshold')
parser.add_argument('--method_name', default='faceboxes', type=str, help='method name')

# parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
# parser.add_argument('--cpu', default=False, type=bool, help='Use cpu nms')
# parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')

# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def _read_pred(pred_dir, gt_dir, silent):
    gt_data = sio.loadmat(gt_dir)
    event_num = 61
    pred_list = [None] * event_num
    for i in range(event_num):
        logger.debug('Read prediction: current event {:d}'.format(i))
        img_list = gt_data['file_list'][i][0]
        img_num = img_list.shape[0]
        bbx_list = [None] * img_num

        for j in range(img_num):
            try:
                with open(
                        '{:s}/{:s}/{:s}.txt'.format(
                            pred_dir, gt_data['event_list'][i][0][0],
                            img_list[j][0][0]), 'r') as f:
                    tmp = f.readlines()
                tmp = [x.strip() for x in tmp]
                bbx_num = int(tmp[1])
                bbx = np.zeros((bbx_num, 5))
                for k in range(bbx_num):
                    raw_info = list(map(lambda x: float(x), tmp[k + 2].split()))
                    bbx[k] = raw_info
                bbx_list[j] = bbx[bbx[:, -1].argsort()[::-1]]
            except:
                logger.error(
                    'Fail to parse the prediction file {:s} {:s}'.format(
                        gt_data['event_list'][i][0][0], img_list[j][0][0]))
        pred_list[i] = bbx_list
    return pred_list


def _norm_score(org_pred_list):
    event_num = 61
    norm_pred_list = [None] * event_num
    max_score = 0.
    min_score = np.inf
    for i in range(event_num):
        pred_list_i = org_pred_list[i]
        max_score = max(max_score, np.max(np.vstack(pred_list_i)[:, -1]))
        min_score = min(min_score, np.min(np.vstack(pred_list_i)[:, -1]))
    for i in range(event_num):
        pred_list_i = copy.copy(org_pred_list[i])
        for j in range(len(pred_list_i)):
            pred_list_i[j][:, -1] -= min_score
            pred_list_i[j][:, -1] /= (max_score - min_score)
        norm_pred_list[i] = pred_list_i
    return norm_pred_list


def _boxoverlap(a, b):
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    inter = w * h
    aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    overlap = aarea + barea - inter
    overlap[overlap == 0] = np.inf
    o = inter / overlap
    o[w <= 0] = 0
    o[h <= 0] = 0
    return o


def _image_evaluation(pred_info, gt_bbx, ignore, IoU_thresh, mimic_eval_bug):
    pred_recall = np.zeros((pred_info.shape[0], 1))
    recall_list = np.zeros((gt_bbx.shape[0], 1))
    proposal_list = np.zeros((pred_info.shape[0], 1))
    proposal_list += 1
    pred_info[:, 2] = pred_info[:, 2] + pred_info[:, 0]
    pred_info[:, 3] = pred_info[:, 3] + pred_info[:, 1]
    gt_bbx[:, 2] = gt_bbx[:, 2] + gt_bbx[:, 0]
    gt_bbx[:, 3] = gt_bbx[:, 3] + gt_bbx[:, 1]
    for h in range(pred_info.shape[0]):
        overlap_list = _boxoverlap(gt_bbx, pred_info[h, :4])
        # TODO: Seems to be a bug in the original code
        if mimic_eval_bug:
            overlap_list = np.array([round(_) for _ in overlap_list])
        max_overlap, idx = np.max(overlap_list), np.argmax(overlap_list)
        if max_overlap >= IoU_thresh:
            if ignore[idx] == 0:
                recall_list[idx] = -1
                proposal_list[h] = -1
            elif recall_list[idx] == 0:
                recall_list[idx] = 1
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def _image_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    img_pr_info = np.zeros((thresh_num, 2))
    for t in range(thresh_num):
        thresh = 1 - (t + 1.) / thresh_num
        try:
            r_index = np.where(pred_info[:, -1] >= thresh)[0][-1]
        except:
            r_index = None
        if r_index is None:
            img_pr_info[t, :] = [0, 0]
        else:
            p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
            img_pr_info[t, :] = [len(p_index), pred_recall[r_index, 0]]
    return img_pr_info


def _dataset_pr_info(thresh_num, org_pr_curve, count_face):
    pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        pr_curve[i, :] = [
            org_pr_curve[i, 1] / org_pr_curve[i, 0],
            org_pr_curve[i, 1] / count_face
        ]
    return pr_curve


def _VOCap(rec, prec):
    mrec = np.hstack([0, rec, 1])
    mpre = np.hstack([0, prec, 0])
    for i in range(mpre.shape[0] - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def _evaluation(data):
    norm_pred_list, gt_dir, setting_name, silent, mimic_eval_bug, IoU_thresh = data[
                                                                                   'norm_pred_list'], data['gt_dir'], \
                                                                               data['setting_name'], data[
                                                                                   'silent'], data['mimic_eval_bug'], \
                                                                               data['IoU_thresh']
    gt_data = sio.loadmat(gt_dir)
    event_num = 61
    thresh_num = 1000
    org_pr_curve = np.zeros((thresh_num, 2))
    count_face = 0
    img_list = np.vstack([_[0] for _ in gt_data['file_list']])
    gt_bbx_list = np.vstack([_[0] for _ in gt_data['face_bbx_list']])
    pred_list = reduce(lambda x, y: x + y, norm_pred_list)
    sub_gt_list = np.vstack([_[0] for _ in gt_data['gt_list']])
    img_pr_info_list = [None] * img_list.shape[0]
    for j in range(img_list.shape[0]):
        gt_bbx = copy.copy(gt_bbx_list[j][0])
        pred_info = copy.copy(pred_list[j])
        keep_index = sub_gt_list[j][0] - 1
        count_face += keep_index.shape[0]
        if gt_bbx.size == 0 or pred_info.size == 0:
            continue
        ignore = np.zeros((gt_bbx.shape[0], 1))
        if keep_index.size > 0:
            ignore[keep_index] = 1
        pred_recall, proposal_list = _image_evaluation(
            pred_info, gt_bbx, ignore, IoU_thresh, mimic_eval_bug)
        img_pr_info = _image_pr_info(thresh_num, pred_info, proposal_list,
                                     pred_recall)
        img_pr_info_list[j] = img_pr_info
    for j in range(img_list.shape[0]):
        img_pr_info = img_pr_info_list[j]
        if img_pr_info is not None:
            org_pr_curve += img_pr_info
    pr_curve = _dataset_pr_info(thresh_num, org_pr_curve, count_face)
    return pr_curve


def wider_eval(pred_dir,
               gt_dir_base,
               silent=True,
               parallel=True,
               mimic_eval_bug=True,
               IoU_thresh=0.5):
    gt_dir = '{:s}/wider_face_val.mat'.format(gt_dir_base)
    pred_list = _read_pred(pred_dir, gt_dir, silent)
    norm_pred_list = _norm_score(pred_list)
    setting_name_list = ['easy_val', 'medium_val', 'hard_val']
    #     setting_name_list = ['hard_val']
    pr_curve = [None] * len(setting_name_list)
    ap = [None] * len(setting_name_list)
    if parallel:
        pool = multiprocessing.Pool(3)
        pr_curve = pool.map(_evaluation, [{
            'norm_pred_list':
                norm_pred_list,
            'gt_dir':
                '{:s}/wider_{:s}.mat'.format(gt_dir_base, setting_name_list[i]),
            'setting_name':
                setting_name_list[i],
            'silent':
                silent,
            'mimic_eval_bug':
                mimic_eval_bug,
            'IoU_thresh':
                IoU_thresh
        } for i in range(3)])
    else:
        for i in range(3):
            pr_curve[i] = _evaluation({
                'norm_pred_list':
                    norm_pred_list,
                'gt_dir':
                    '{:s}/wider_{:s}.mat'.format(gt_dir_base,
                                                 setting_name_list[i]),
                'setting_name':
                    setting_name_list[i],
                'silent':
                    silent,
                'mimic_eval_bug':
                    mimic_eval_bug,
                'IoU_thresh':
                    IoU_thresh
            })
    for i in range(len(setting_name_list)):
        ap[i] = _VOCap(pr_curve[i][:, 1], pr_curve[i][:, 0])
    return ap, pr_curve


def evaluate_detections(
        all_boxes,
        output_dir=args.save_dir,
        method_name=args.method_name,
        step=0):
    logger.info('Evaluating detections using unofficial WIDER toolbox...')
    ap, pr = wider_eval(
        args.det_dir,
        args.gt_dir,
        mimic_eval_bug=True,
        IoU_thresh=args.iou_threshold)
    #     with tarfile.open(os.path.join(output_dir, 'result.tar.gz'),
    #                       'w:gz') as tar:
    #         tar.add(
    #             detections_txt_path,
    #             arcname=os.path.basename(detections_txt_path))
    #     shutil.rmtree(detections_txt_path)

    result = 'Easy: {:.4f}, Medium: {:.4f}, Hard: {:.4f}'.format(*ap)
    print(result)
    return result


if __name__ == "__main__":

    ap, pr = wider_eval(
        args.det_dir,
        args.gt_dir,
        mimic_eval_bug=True,
        IoU_thresh=args.iou_threshold)
    result = 'Easy: {:.4f}, Medium: {:.4f}, Hard: {:.4f}'.format(*ap)
    np.save("%s/easy_precision_recall_iou%.3f.npy" % (args.save_dir, args.iou_threshold), pr[0])
    np.save("%s/medium_precision_recall_iou%.3f.npy" % (args.save_dir, args.iou_threshold), pr[1])
    np.save("%s/hard_precision_recall_iou%.3f.npy" % (args.save_dir, args.iou_threshold), pr[2])

    plt.figure(figsize=(8, 7))
    for mean_p, mean_r in pr[0]:
        if mean_p is not None:
            plt.plot(mean_p, mean_r, 'o-')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.title('mobileface easy set ROC curve')
    plt.savefig("{}/{}_easy_set.png".format(args.save_dir, args.method_name))

    plt.figure(figsize=(8, 7))
    for mean_p, mean_r in pr[1]:
        if mean_p is not None:
            plt.plot(mean_p, mean_r, 'o-')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.title('mobileface medium set ROC curve')
    plt.savefig("{}/{}_medium_set.png".format(args.save_dir, args.method_name))

    plt.figure(figsize=(8, 7))
    for mean_p, mean_r in pr[2]:
        if mean_p is not None:
            plt.plot(mean_p, mean_r, 'o-')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.title('mobileface hard set ROC curve')
    plt.savefig("{}/{}_hard_set.png".format(args.save_dir, args.method_name))
    print(result)
