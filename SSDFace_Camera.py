"""
    By.Wheat
    2020.05.21
"""
from dataset import *
from common import *

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

net = SSDType(VGG(3))
# net.auto_load_weights(path.join(WEIGHT_ROOT, net.name + '_' + 'FDDB_fromtrain' + '.pth'))
net.auto_load_weights(path.join(WEIGHT_ROOT, net.name + '_' + 'WIDER' + '_20000.pth'))

pre_solve = BaseTransform(net.size, (104.0, 117.0, 123.0))

use_pylon = False
if use_pylon:
    from pypylon import pylon

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
else:
    camera = cv2.VideoCapture(0)

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
    x = x[:, :, (2, 1, 0)]

    x = torch.from_numpy(x).permute(2, 0, 1)
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    y = net.detect(x, conf_thresh=0.3)
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

    image = cv2.resize(image, (800, 600))
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

