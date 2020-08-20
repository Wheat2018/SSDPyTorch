import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import math
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from FlashNet.facedet.models.common_ops import *
from FlashNet.facedet.utils.ops.nms.nms_wrapper import nms
import cv2

__all__ = ['FCOSFace']

class FCOSFace(nn.Module):

    def __init__(self, phase, cfg):
        super(FCOSFace, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self._stride = cfg['strides']
        self._stages = len(self._stride)

        self.conv1 = conv_bn(3, 8, 2)
        # self.conv1 = conv_bn_5X5(3, 12, 2)
        # self.conv2 = conv_bn(12, 12, 2)
        # self.conv1 = hetconv_bn_5X5(3, 12, num_groups=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.p3_conv1 = InvertedResidual(inp=8, oup=24, kernel=3, stride=2, expand_ratio=4)
        self.p3_conv2 = InvertedResidual(inp=24, oup=24, kernel=5, stride=1, expand_ratio=4)
        self.p3_conv3 = InvertedResidual(inp=24, oup=32, kernel=5, stride=1, expand_ratio=4)

        self.p4_conv1 = InvertedResidual(inp=32, oup=48, kernel=3, stride=2, expand_ratio=4)
        self.p4_conv2 = InvertedResidual(inp=48, oup=48, kernel=5, stride=1, expand_ratio=3)
        self.p4_conv3 = InvertedResidual(inp=48, oup=48, kernel=3, stride=1, expand_ratio=3)

        self.p5_conv1 = InvertedResidual(inp=48, oup=64, kernel=3, stride=2, expand_ratio=3)
        self.p5_conv2 = InvertedResidual(inp=64, oup=64, kernel=5, stride=1, expand_ratio=3)
        self.p5_conv3 = InvertedResidual(inp=64, oup=64, kernel=3, stride=1, expand_ratio=3)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(48, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.box_head, self.ldmk_head, self.cls_head = self.loss_head()

    def loss_head(self):

        box_heads = list()
        ldmk_heads = list()
        cls_heads = list()

        box_heads += [InvertedResidual_Head(inp=32, oup=4, kernel=3, stride=1, expand_ratio=1)]
        box_heads += [InvertedResidual_Head(inp=32, oup=4, kernel=3, stride=1, expand_ratio=1)]
        box_heads += [InvertedResidual_Head(inp=32, oup=4, kernel=3, stride=1, expand_ratio=1)]

        ldmk_heads += [InvertedResidual_Head(inp=32, oup=10, kernel=3, stride=1, expand_ratio=1)]
        ldmk_heads += [InvertedResidual_Head(inp=32, oup=10, kernel=3, stride=1, expand_ratio=1)]
        ldmk_heads += [InvertedResidual_Head(inp=32, oup=10, kernel=3, stride=1, expand_ratio=1)]

        cls_heads += [InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)]
        cls_heads += [InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)]
        cls_heads += [InvertedResidual_Head(inp=32, oup=1, kernel=3, stride=1, expand_ratio=1)]
        return nn.Sequential(*box_heads), nn.Sequential(*ldmk_heads), nn.Sequential(*cls_heads)

    def _upsample_add(self, x, y):
        size = [v for v in y.size()[2:]]
        size = [int(i) for i in size]
        return F.interpolate(x, size=size, mode='nearest') + y

    def initialize(self, pre_trained):
        if pre_trained:
            # Initialize using weights from pre-trained model
            if not os.path.isfile(pre_trained):
                raise ValueError('No checkpoint {}'.format(pre_trained))

            print('Fine-tuning weights from {}...'.format(os.path.basename(pre_trained)))
            state_dict = self.state_dict()
            chk = torch.load(pre_trained, map_location=lambda storage, loc: storage)
            ignored = ['cls_head.8.bias', 'cls_head.8.weight']
            weights = {k: v for k, v in chk['state_dict'].items() if k not in ignored}
            state_dict.update(weights)
            self.load_state_dict(state_dict)

            del chk, weights
            torch.cuda.empty_cache()

        else:
            # Initialize backbones(s)
            for _, backbone in self.backbones.items():
                backbone.initialize()

            # Initialize heads
            def initialize_layer(layer):
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, val=0)

            self.cls_head.apply(initialize_layer)
            self.box_head.apply(initialize_layer)
            self.ldmk_head.apply(initialize_layer)

    def forward(self, x):
        features = list()
        box = list()
        ldmk = list()
        cls = list()
        batchsize = x.size(0)
        x = self.conv1(x)
        if self._stride[0] == 8:
            s4 = self.maxpool1(x)
            x = self.p3_conv1(s4)
            x = self.p3_conv2(x)
            s8 = self.p3_conv3(x)
            x = self.p4_conv1(s8)
            x = self.p4_conv2(x)
            s16 = self.p4_conv3(x)
            x = self.p5_conv1(s16)
            x = self.p5_conv2(x)
            s32 = self.p5_conv3(x)

            p5 = self.latlayer1(s32)
            p4 = self._upsample_add(p5, self.latlayer2(s16))
            p3 = self._upsample_add(p4, self.latlayer3(s8))

            features += [p3]
            features += [p4]
            features += [p5]
        elif self._stride[0] == 4:
            x = self.p3_conv1(x)
            x = self.p3_conv2(x)
            s4 = self.p3_conv3(x)
            x = self.p4_conv1(s4)
            x = self.p4_conv2(x)
            s8 = self.p4_conv3(x)
            x = self.p5_conv1(s8)
            x = self.p5_conv2(x)
            s16 = self.p5_conv3(x)

            p4 = self.latlayer1(s16)
            p3 = self._upsample_add(p4, self.latlayer2(s8))
            p2 = self._upsample_add(p3, self.latlayer3(s4))

            features += [p4]
            features += [p3]
            features += [p2]


        for idx, t in enumerate(features):
            # print(idx)
            #bchw -> bhwc
            box.append(self.box_head[idx](t).permute(0, 2, 3, 1).contiguous())
            ldmk.append(self.ldmk_head[idx](t).permute(0, 2, 3, 1).contiguous())
            cls.append(self.cls_head[idx](t).permute(0, 2, 3, 1).contiguous())

        box = torch.cat([o.view(o.size(0), -1) for o in box], 1)
        ldmk = torch.cat([o.view(o.size(0), -1) for o in ldmk], 1)
        cls = torch.cat([o.view(o.size(0), -1) for o in cls], 1)

        # if self.phase == "test":
        #     wh = wh.exp()
        output = (box.view(batchsize, -1, 4),
                  2 * ldmk.view(batchsize, -1, 10).sigmoid() - 1,
                  cls.view(batchsize, -1).sigmoid())

        return output

    def post_process(self,
                     foward_output,
                     im_height,
                     im_width,
                     resize,
                     use_cuda=True,
                     confidence_threshold=0.1,
                     nms_threshold=0.3,
                     top_k=5000,
                     keep_top_k=750):
        if self.use_ldmk:
            wh, conf, _, ldmk = foward_output
        else:
            wh, conf, _ = foward_output

        box_cords = self.generate_box_cords(im_height, im_width)
        if use_cuda:
            box_cords = box_cords.cuda()

        if self.use_ldmk:
            boxes, ldmks = self.decode(box_cords.unsqueeze(0), wh.exp(), ldmk)
        else:
            boxes = self.decode(box_cords.unsqueeze(0), wh.exp())

        boxes = boxes / resize
        boxes = boxes.data.cpu().numpy()
        # scores: one dim
        scores = conf.data.cpu().numpy()[0, :, 0]
        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]
        # import pdb
        # pdb.set_trace()
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        dets, nms_idx = nms(dets, nms_threshold)
        # keep top-K after NMS
        dets = dets[:keep_top_k, :]

        if self.use_ldmk:
            ldmks = ldmks / resize
            ldmks = ldmks.data.cpu().numpy()
            ldmks = ldmks[inds]
            ldmks = ldmks[order]
            ldmks = ldmks[nms_idx]
            ldmks = ldmks[:keep_top_k, :]
            return dets, ldmks
        else:
            return dets

    def decode(self, box_cord, box_deltas, ldmk_deltas=None):
        """Decode landmark predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            box_deltas (tensor):  Shape: [N,2]   format: [w, h]
            ldmk_deltas (tensor):  Shape: [N,10]   format: [x1,y1,...,x5,y5]

        Return:
            decoded boxes: [N, 4]
            decoded landmark: [N, 10]
        """
        # [B, N, 4]->[B, N, 1]
        factor = box_cord[:, :, 2].unsqueeze(-1)
        factor = torch.cat([factor, factor], dim=-1)
        box_pred = box_deltas * factor
        width = box_pred[:, :, 0:1]
        height = box_pred[:, :, 1:2]

        # [B, N, 2]->[B, N, 1]
        cx = box_cord[:, :, 0:1]
        cy = box_cord[:, :, 1:2]
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        boxes = torch.cat([x1, y1, x2, y2], dim=-1).squeeze(0)

        if ldmk_deltas is not None:
            # import ipdb
            # ipdb.set_trace()
            ldmk_deltas = ldmk_deltas.squeeze(0)
            priors = torch.cat([cx, cy, width, height], dim=-1).squeeze(0)
            for i in range(0, 5):
                ldmk_deltas[:, 2 * i] = ldmk_deltas[:, 2 * i] * priors[:, 2] + priors[:, 0]
                ldmk_deltas[:, 2 * i + 1] = ldmk_deltas[:, 2 * i + 1] * priors[:, 3] + priors[:, 1]
            return boxes, ldmk_deltas
        else:
            return boxes


    def generate_box_cords(self, img_height, img_width):
        """
        Args:
            img : [H, W, 3]
            boxes : [N, 5]
        Return:
            cls_targets: [所有步长的特征图点的个数之和, num_classes]
            ctr_targets: [所有步长的特征图点的个数之和, 1]
            box_targets: [所有步长的特征图点的个数之和, 4]
            cor_targets: [所有步长的特征图点的个数之和, 2]  特征图的点对应于原图的坐标
        """
        cor_targets = []
        for i in range(self._stages):
            stride = self._stride[i]
            fw, fh = img_width, img_height
            while stride > 1:
                fw = int(np.ceil(fw / 2))
                fh = int(np.ceil(fh / 2))
                stride /= 2

            rx = torch.arange(0, fw).view(1, -1)
            ry = torch.arange(0, fh).view(-1, 1)
            sx = rx.repeat(fh, 1).float()
            sy = ry.repeat(1, fw).float()
            syx = torch.stack((sy.view(-1), sx.view(-1))).transpose(1, 0).long()
            by = syx[:, 0] * self._stride[i] + self._stride[i] / 2
            bx = syx[:, 1] * self._stride[i] + self._stride[i] / 2
            #            by = syx[:, 0] * self._stride[i]
            #            bx = syx[:, 1] * self._stride[i]
            by = by.clamp(0, img_height - 1)
            bx = bx.clamp(0, img_width - 1)
            stride = torch.ones_like(bx)*self._stride[i]
            # (fh*fw, 2)
            cor_targets.append(torch.stack((bx, by, stride), dim=1))
        cor_targets = torch.cat([cor_target for cor_target in cor_targets], dim=0)
        return cor_targets.float()


def draw_results(img_origin, img_save_name, bboxes_scores, ldmks):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        image_path (string): file name.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    for idx, box_score in enumerate(bboxes_scores):
        xmin, ymin, xmax, ymax, score = box_score
        if score > 0.4:
            if xmax - xmin < 25 and ymax - ymin < 25:
                cv2.rectangle(img_origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
                cv2.putText(img_origin, str('%0.2f' % score), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0),
                            1)
            elif xmax - xmin < 35 and ymax - ymin < 35:
                cv2.rectangle(img_origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 255), 2)
                cv2.putText(img_origin, str('%0.2f' % score), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 255), 1)
            else:
                cv2.rectangle(img_origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                cv2.putText(img_origin, str('%0.2f' % score), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

            landmark = ldmks[idx]
            landmark = landmark.reshape(5, 2)
            for point in landmark:
                cv2.circle(img_origin, (int(point[0]), int(point[1])), radius=1, color=(0, 0, 255), thickness=2)

    cv2.imwrite(img_save_name, img_origin)

if __name__=="__main__":

    from facedet.utils.misc.checkpoint import *
    import torch.backends.cudnn as cudnn
    from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number
    net_cfg = {
        'net_name': 'CenterFace',
        'num_classes': 1,
        'use_ldmk': True,
        'ldmk_reg_type': 'coord'
    }

    if net_cfg["use_ldmk"]:
        # img_name = "13_Interview_Interview_2_People_Visible_13_167.jpg"
        img_name = './speed_benchmark/0_Parade_Parade_0_470.jpg'
        trained_model = "/mnt/lustre/geyongtao/RetinaFace.PyTorch-sz/weights/centerface_ldmk_coord/AdamW/CenterFace_epoch_220.pth"
        img_save_name = "0_Parade_Parade_0_470_result_220.jpg"
        # img_save_name = "13_Interview_Interview_2_People_Visible_13_167_result.jpg"
    else:
        img_name = "/mnt/lustre/geyongtao/RetinaFace.PyTorch-sh/1_Handshaking_Handshaking_1_579.jpg"
        trained_model = "/mnt/lustre/geyongtao/RetinaFace.PyTorch-sh/weights/centerface/AdamW/CenterFace_epoch_295.pth"
        img_save_name = "1_Handshaking_Handshaking_1_579_result_epoch.jpg"

    use_cuda = True
    img_origin = np.float32(cv2.imread(img_name, cv2.IMREAD_COLOR))


    net = CenterFace(phase='test', cfg=net_cfg)
    # flops and params estimation
    img_dim = 640
    input_size = (1, 3, img_dim, img_dim)
    img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])
    net = add_flops_counting_methods(net)
    net.start_flops_count()
    feat = net(img)
    flops = net.compute_average_flops_cost()
    print('Net Flops:  {}'.format(flops_to_string(flops)))
    print('Net Params: ' + get_model_parameters_number(net))
    # load model
    net = load_model(net, trained_model)
    net.eval()
    print('Finished loading model!')
    # preprocess image
    resize = 1600 / img_origin.shape[0]
    img = cv2.resize(img_origin, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    if use_cuda:
        img = img.cuda()
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # foward
    with torch.no_grad():
        forward_outputs = net(img)  # forward pass
        if net_cfg["use_ldmk"]:
            dets, ldmks = net.post_process(forward_outputs,
                                    im_height,
                                    im_width,
                                    resize)
            draw_results(img_origin, img_save_name, dets, ldmks)
        else:
            dets = net.post_process(forward_outputs,
                                    im_height,
                                    im_width,
                                    resize)
            draw_results(img_origin, img_save_name, dets)




