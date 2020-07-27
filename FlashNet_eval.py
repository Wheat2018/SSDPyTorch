"""
    By.Wheat
    2020.07.26
"""

from dataset import *
from FlashNet.models.centerface import *
from FlashNet.utils.misc.checkpoint import *

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

eval_save_folder = 'face_eval/wider'
os.makedirs(eval_save_folder, exist_ok=True)

net_cfg = {
    'net_name': 'CenterFace',
    'num_classes': 1,
    'use_ldmk': False
}

net = CenterFace(phase='test', cfg=net_cfg)
net = load_model(net, "FlashNet/checkpoints/CenterFace.pth")
net.eval()
dataset = WIDER(dataset='val',
                image_enhancement_fn=BaseTransform((-1, 600), (104.0, 117.0, 123.0)))

t0 = time.time()
t00 = t0
for idx in range(len(dataset)):
    x, boxes, h, w = dataset.pull_item(idx)
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
        net = net.cuda()
    wh, conf, _ = net(x)  # forward pass
    detection = net.post_process(wh,
                                 conf,
                                 x.shape[-2],
                                 x.shape[-1],
                                 h,
                                 w)
    detection = detection[:, (4, 0, 1, 2, 3)]
    detection[:, 1] /= w
    detection[:, 3] /= w
    detection[:, 2] /= h
    detection[:, 4] /= h
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
