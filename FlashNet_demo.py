"""
    By.Wheat
    2020.08.21
"""

from dataset import *
from matplotlib import pyplot as plt
from FlashNet.facedet.models.flashnet import FlashNet
import FlashNet.facedet.configs.flashnet_1024_2_anchor as fl_cfg
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
from FlashNet.facedet.losses.multibox_loss import MultiBoxLoss

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from FlashNet_train import preproc


def detect(out, priors, variance, top_k=200, conf_thresh=0.1, nms_thresh=0.45):

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
        # For each class, perform nms
        conf_scores = conf_data[i].clone().t()      # [classes_num, priors_num]

        output_each = torch.Tensor()
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
            ids, count = nms(boxes, scores, nms_thresh, top_k)
            output_cl = torch.cat((scores[ids[:count]].unsqueeze(1),
                                 boxes[ids[:count]]), 1)
            if classes_num > 2:
                output_cl = torch.cat((torch.Tensor([cl]).expand(count, 1),
                                       output_cl), 1)
            output_each = torch.cat((output_each, output_cl), 0)
        output += [output_each]
    return output


cfg = {
    'net_cfg': fl_cfg.net_cfg,
    'anchor_cfg': fl_cfg.anchor_cfg,
    'train_cfg': fl_cfg.train_cfg,
    'test_cfg': fl_cfg.test_cfg
}
# cfg = Config.fromfile('./FlashNet/facedet/configs/flashnet_1024_2_anchor.py')

rgb_means = (104, 117, 123)
img_dim = cfg['train_cfg']['input_size']

net = FlashNet(phase='test', cfg=cfg['net_cfg'])

# testset = WIDER(dataset='val', image_enhancement_fn=BaseTransform((0, 0), (104.0, 117.0, 123.0)))
testset = FDDB(dataset='test', image_enhancement_fn=BaseTransform((0, 0), (104.0, 117.0, 123.0)))
# testset = FDDB(dataset='test', image_enhancement_fn=AugmentationCall(preproc(img_dim, rgb_means)))

load_weights(net, path.join(WEIGHT_ROOT, 'FlashNet_' + testset.name + '.pth'))
# net.auto_load_weights(path.join(PRETRAIN_ROOT, 'vgg16_reducedfc.pth'))
criterion = MultiBoxLoss(2, 0.35, True, 0, True, 3, 0.35, False, cfg['train_cfg']['use_ldmk'])

for img_id in range(len(testset)):
    image = testset.pull_image(img_id)
    x, gt_boxes, h, w = testset.pull_item(img_id)

    t1 = time.time()
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    # out = net(x)
    # loss_l, loss_c = criterion(out, [torch.Tensor(gt_boxes)])
    # loss = loss_l + loss_c
    out = net(x)
    # priors = PriorBox(cfg['anchor_cfg']).forward()
    priors = PriorBox(cfg['anchor_cfg'], image_size=(h, w), phase='test').forward()

    loss_l, loss_c = criterion(out, priors, torch.FloatTensor(gt_boxes).unsqueeze(0).cuda())
    loss = cfg['train_cfg']['loc_weight'] * loss_l + cfg['train_cfg']['cls_weight'] * loss_c
    print('iter ' + repr(img_id) +
          '\t|| loc/conf/all: %.4f / %.4f / %.4f ||' % (loss_l.item(), loss_c.item(), loss.item()),
          end=' ')

    y = detect(out, priors, cfg['anchor_cfg']['variance'])
    detection = y[0]
    t2 = time.time()
    # print(str(img_id), '\tloss: %.4f %.4f %.4f %.4f sec' % (float(loss_l), float(loss_c), float(loss), t2-t1))
    print(str(img_id), '\t%.4f sec' % (t2-t1))
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)

    cv_flag = True
    if cv_flag:
        color = [0, 255, 0]
        # scale each detection back up to the image
        for i in range(gt_boxes.shape[0]):
            pt = (torch.Tensor(gt_boxes[i][:4]) * scale).type(torch.int32).cpu().numpy()
            cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), [0, 255, 0], 2)
        color = [0, 0, 255]
        for i in range(detection.shape[0]):
            score = detection[i, 0]

            display_txt = '%.2f' % score
            pt = (detection[i, 1:] * scale).type(torch.int32).cpu().numpy()

            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
            cv2.fillPoly(image,
                         np.array([[[pt[0], pt[1]], [pt[0]+25, pt[1]], [pt[0]+25, pt[1]+15], [pt[0], pt[1]+15]]]),
                         color)
            inverse_color = [255 - x for x in color]
            cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, inverse_color, lineType=cv2.LINE_AA)
        cv2.imshow('test', image)
        k = cv2.waitKey(0)
        if k == 27:
            break
    else:
        plt.figure(figsize=(10, 10))
        currentAxis = plt.gca()
        color = [0, 1, 0, 1.0]
        # scale each detection back up to the image
        for i in range(gt_boxes.shape[0]):
            pt = (torch.Tensor(gt_boxes[i][:4]) * scale).type(torch.int32).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        color = [1, 0, 0, 1.0]
        for i in range(detection.shape[0]):
            score = detection[i, 0]

            display_txt = '%.2f' % score
            pt = (detection[i, 1:]*scale).type(torch.int32).cpu().numpy()

            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.show()
        plt.savefig('test.jpg')
        k = input()
        if k == 27:
            break
