"""
    By.Wheat
    2020.07.26
"""

from dataset import *
from common import *
from matplotlib import pyplot as plt
from FlashNet.facedet.models.centerface import *
from FlashNet.facedet.utils.misc.checkpoint import *
from mmcv import  Config

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

net_cfg = {
    'net_name': 'CenterFace',
    'num_classes': 1,
    'use_ldmk': False,
    'ldmk_reg_type': None
}
# cfg = Config.fromfile('./FlashNet/facedet/configs/deprecated/centerface.py')

net = CenterFace(phase='test', cfg=net_cfg)
net = load_model(net, "./FlashNet/facedet/checkpoints/CenterFace.pth")
net.eval()

im_height = 1600
testset = WIDER(dataset='val', image_enhancement_fn=BaseTransform((-1, im_height), (104.0, 117.0, 123.0)))
# testset = FDDB(dataset='test', image_enhancement_fn=BaseTransform((-1, 600), (104.0, 117.0, 123.0)))

for img_id in range(len(testset)):
    image = testset.pull_image(img_id)
    x, gt_boxes, h, w = testset.pull_item(img_id)

    t1 = time.time()
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
        net = net.cuda()
    with torch.no_grad():
        forward = net(x)  # forward pass
    detection = net.post_process(forward,
                                 x.shape[-2],
                                 x.shape[-1],
                                 im_height / h,
                                 confidence_threshold=0.4)
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
            score = detection[i, 4]

            display_txt = '%.2f' % score
            pt = detection[i, :4].astype(np.int)

            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
            # cv2.fillPoly(image,
            #              np.array([[[pt[0], pt[1]], [pt[0]+25, pt[1]], [pt[0]+25, pt[1]+15], [pt[0], pt[1]+15]]]),
            #              color)
            # inverse_color = [255 - x for x in color]
            # cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, inverse_color, lineType=cv2.LINE_AA)
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
            score = detection[i, 4]

            display_txt = '%.2f' % score
            pt = detection[i, :4].astype(np.int)

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


