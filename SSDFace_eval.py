"""
    By.Wheat
    2020.05.14
"""

from dataset import *

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

eval_save_folder = 'face_eval/'
if not os.path.exists(eval_save_folder):
    os.mkdir(eval_save_folder)

net = SSDType(VGG(3))
dataset = FDDB(dataset='all',
               image_enhancement_fn=BaseTransform(300, (104.0, 117.0, 123.0)))
net.auto_load_weights(path.join(WEIGHT_ROOT, net.name + '_' + dataset.name + '.pth'))

t0 = time.time()
for idx in range(len(dataset)):
    x, boxes, h, w = dataset.pull_item(idx)
    x = x.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    y = net.detect(x)
    detection = y[0]
    dataset.sign_item(idx, detection, h, w)
    if idx % 10 == 0:
        print('detect:%d/%d' % (idx, len(dataset)))
t1 = time.time()
print('detect cost: %.4f sec, FPS: %.4f' % ((t1 - t0), len(dataset) / (t1 - t0)))

filenames = dataset.write_eval_result(eval_save_folder)
t2 = time.time()
print('write cost: %.4f sec' % (t2 - t1))
print('have written:')
for filename in filenames:
    print(filename)
