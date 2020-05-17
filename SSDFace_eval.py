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
evalset = FDDB(dataset='all',
               image_enhancement_fn=BaseTransform(300, (104.0, 117.0, 123.0)))
net.auto_load_weights(path.join(WEIGHT_ROOT, net.name + '_' + evalset.name + '.pth'))

range_of_each_fold = evalset.num_of_each_fold
for i in range(len(range_of_each_fold) - 1):
    range_of_each_fold[i + 1] += range_of_each_fold[i]
range_of_each_fold = [0] + range_of_each_fold

FPSs = []
for fold in range(len(range_of_each_fold) - 1):
    save_filename = eval_save_folder + 'fold-%s-out.txt' % FDDB.image_folds['all'][fold]
    all_boxes = []
    t0 = time.time()
    for idx in range(range_of_each_fold[fold], range_of_each_fold[fold + 1]):
        x, gt_boxes, h, w = evalset.pull_item(idx)

        x = x.unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()
        y = net.detect(x)
        detection = y[0]

        # scale each detection back up to the image
        scale = torch.Tensor([w, h, w, h])

        boxes = []
        for i in range(detection.shape[0]):
            box = torch.Tensor(detection[i, 1:]) * scale
            box = torch.Tensor([box[0], box[1],
                                box[2] - box[0],
                                box[3] - box[1],
                                detection[i, 0]])
            boxes.append(box)

        all_boxes.append(boxes)
        if idx % 10 == 0:
            print('detect:%d/%d' % (idx, len(evalset)))
    t1 = time.time()
    print('detect cost: %.4f sec, FPS: %.4f' % ((t1 - t0), len(all_boxes) / (t1 - t0)))
    FPSs += [len(all_boxes) / (t1 - t0)]

    with open(save_filename, 'wt') as file:
        for idx in range(range_of_each_fold[fold], range_of_each_fold[fold + 1]):
            image_name = evalset.image_names[idx]
            image_name, _ = os.path.splitext(image_name)

            idx_from_zero = idx - range_of_each_fold[fold]
            file.write(image_name + '\n')
            file.write(str(len(all_boxes[idx_from_zero])) + '\n')
            for box in all_boxes[idx_from_zero]:
                line = ''
                for e in box:
                    line += '%.6f ' % e
                file.write(line + '\n')
    t2 = time.time()
    print('write cost: %.4f sec' % (t2 - t1))
    print('have written ' + save_filename)

print('average FPS: %.4f sec' % (sum(FPSs) / len(FPSs)))

