"""
    By.Wheat
    2020.09.02
"""

from dataset import *
from FlashNet.facedet.models.flashnet import FlashNet
from FlashNet.facedet.utils.anchor.prior_box import PriorBox
from FlashNet.facedet.utils.bbox.box_utils import decode
from mmcv import Config

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

eval_save_folder = 'face_eval/wider'
os.makedirs(eval_save_folder, exist_ok=True)


def detect(out, priors, variance, top_k=5000, conf_thresh=0.4, nms_thresh=0.3):

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


cfg = Config.fromfile('./FlashNet/facedet/configs/flashnet_1024_2_anchor.py')

rgb_means = (104, 117, 123)
img_dim = cfg['train_cfg']['input_size']

net = FlashNet(phase='test', cfg=cfg['net_cfg'])
net.eval()

load_weights(net, path.join(WEIGHT_ROOT, 'FlashNet_' + 'WIDER' + '.pth'))

dataset = WIDER(dataset='val',
                image_enhancement_fn=BaseTransform((0, 1600), (104.0, 117.0, 123.0)))

t0 = time.time()
t00 = t0
for idx in range(len(dataset)):
    x, boxes, h, w = dataset.pull_item(idx)
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        out = net(x)
    priors = PriorBox(cfg['anchor_cfg'], image_size=x.shape[-2:], phase='test').forward()
    y = detect(out, priors, cfg['anchor_cfg']['variance'])
    detection = y[0]
    dataset.sign_item(idx, detection, h, w)
    if idx % 10 == 0:
        print('detect:%d/%d, FPS:%.4f' % (idx, len(dataset), 10 / (time.time() - t00)))
        t00 = time.time()
t1 = time.time()
print('detect cost: %.4f sec, FPS: %.4f' % ((t1 - t0), len(dataset) / (t1 - t0)))

filenames = dataset.write_eval_result(eval_save_folder)
t2 = time.time()
print('write cost: %.4f sec' % (t2 - t1))
print('have written:')
print(len(filenames))
