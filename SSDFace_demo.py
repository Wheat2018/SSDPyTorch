"""
    By.Wheat
    2020.05.16
"""

from dataset import *
from common import *

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = SSDType(VGG(3))
testset = FDDB(dataset='test',
               image_enhancement_fn=BaseTransform(net.size, (104.0, 117.0, 123.0)))
net.auto_load_weights(path.join(WEIGHT_ROOT, net.name + '_' + testset.name + '.pth'))
# net.auto_load_weights(path.join(PRETRAIN_ROOT, 'vgg16_reducedfc.pth'))
criterion = SSDLossType(0.5, 3, variance=net.variance)

for img_id in range(len(testset)):
    image = testset.pull_image(img_id)
    x, gt_boxes, _, _ = testset.pull_item(img_id)

    t1 = time.time()
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    out = net(x)
    loss_l, loss_c = criterion(out, [torch.Tensor(gt_boxes)])
    loss = loss_l + loss_c
    print('loss', float(loss))
    y = net.detect(x)
    detection = y[0]
    t2 = time.time()
    print('%.4f' % (t2-t1))

    color = [0, 0, 255]
    # scale each detection back up to the image
    scale = torch.Tensor(image.shape[1::-1]).repeat(2)
    for i in range(gt_boxes.shape[0]):
        pt = (torch.Tensor(gt_boxes[i][:4]) * scale).type(torch.int32).cpu().numpy()
        cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), [0, 255, 0], 2)

    for i in range(detection.shape[0]):
        score = detection[i, 0]

        display_txt = '%.2f' % score
        pt = (detection[i, 1:]*scale).type(torch.int32).cpu().numpy()

        inverse_color = [255 - x for x in color]
        cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), color, 2)
        cv2.fillPoly(image, np.array([[[pt[0], pt[1]],
                                      [pt[0]+25, pt[1]],
                                      [pt[0]+25, pt[1]+15],
                                      [pt[0], pt[1]+15]]]),
                     color)
        cv2.putText(image, display_txt, (int(pt[0]), int(pt[1]) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, inverse_color, lineType=cv2.LINE_AA)

    cv2.imshow('test', image)
    cv2.waitKey(0)
