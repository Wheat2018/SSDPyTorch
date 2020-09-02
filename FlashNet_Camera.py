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

pre_solve = BaseTransform((0, 600), (104.0, 117.0, 123.0))
load_weights(net, path.join(WEIGHT_ROOT, 'FlashNet_' + 'WIDER' + '.pth'))

use_pylon = False
if use_pylon:
    from pypylon import pylon

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
else:
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

t0 = time.time()
count = 0
while True:
    if use_pylon:
        image = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        image_handle = image
        image = image.Array
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_Y422)
    else:
        success, image = camera.read()

    x, _, _ = pre_solve(image)
    # x = x[:, :, (2, 1, 0)]

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        out = net(x)
    priors = PriorBox(cfg['anchor_cfg'], image_size=x.shape[-2:], phase='test').forward()
    y = detect(out, priors, cfg['anchor_cfg']['variance'])
    detection = y[0]

    color = [0, 255, 0]
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    for i in range(detection.shape[0]):
        score = detection[i, 0]

        display_txt = '%.2f' % score
        pt = (detection[i, 1:] * scale).type(torch.int32).cpu().numpy()

        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
        cv2.fillPoly(image,
                     np.array([[[pt[0], pt[1]], [pt[0] + 25, pt[1]], [pt[0] + 25, pt[1] + 15], [pt[0], pt[1] + 15]]]),
                     color)
        inverse_color = [255 - x for x in color]
        cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, inverse_color, lineType=cv2.LINE_AA)

    cv2.imshow('image', image)

    if use_pylon:
        image_handle.Release()
    count = (count + 1) % 10
    if count == 0:
        print('FPS:%.4f' % (10 / (time.time() - t0)))
        t0 = time.time()
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        break
    elif k == ord("s"):
        cv2.imwrite("image2.jpg", image)
        cv2.destroyAllWindows()
        break

if use_pylon:
    camera.Close()
else:
    camera.release()

