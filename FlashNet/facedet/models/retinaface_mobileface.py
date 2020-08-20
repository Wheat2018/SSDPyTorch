import torch
import torch.nn.functional as F
import torch.nn as nn
from facedet.utils.misc import add_flops_counting_methods, flops_to_string, get_model_parameters_number

class RetinaFace_MobileNet(nn.Module):
    def __init__(self):
        super(RetinaFace_MobileNet, self).__init__()

        self.mobilenet0_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1, groups=8, bias=False),
            nn.BatchNorm2d(num_features=8, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(num_features=16, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv13 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv14 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv15 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv16 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv17 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv18 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv19 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv20 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv21 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv22 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv23 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv24 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv25 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True))

        self.mobilenet0_conv26 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c3_lateral = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c3_det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=32, eps=2e-05, momentum=0.9))

        self.rf_c3_det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c3_det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c3_det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c3_det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c3_det_concat_relu = nn.Sequential(
            nn.ReLU(inplace=True))

        self.face_rpn_cls_score_stride32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True))

        self.face_rpn_cls_score_stride32_softmax = nn.Sequential(
            nn.Softmax(dim=1))

        self.face_rpn_bbox_pred_stride32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True))

        self.face_rpn_landmark_pred_stride32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=20, kernel_size=1, stride=1, padding=0, bias=True))

        self.rf_c2_lateral = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True))

        self.rf_c3_upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.rf_c2_aggr = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c2_det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=32, eps=2e-05, momentum=0.9))

        self.rf_c2_det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c2_det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c2_det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c2_det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c2_det_concat_relu = nn.Sequential(
            nn.ReLU(inplace=True))

        self.face_rpn_cls_score_stride16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True))

        self.face_rpn_cls_score_stride16_softmax = nn.Sequential(
            nn.Softmax(dim=1))

        self.face_rpn_bbox_pred_stride16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True))

        self.face_rpn_landmark_pred_stride16 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=20, kernel_size=1, stride=1, padding=0, bias=True))

        self.rf_c1_red_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c2_upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'))

        self.rf_c1_aggr = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=64, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c1_det_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=32, eps=2e-05, momentum=0.9))

        self.rf_c1_det_context_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c1_det_context_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c1_det_context_conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9),
            nn.ReLU(inplace=True))

        self.rf_c1_det_context_conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_features=16, eps=2e-05, momentum=0.9))

        self.rf_c1_det_concat_relu = nn.Sequential(
            nn.ReLU(inplace=True))

        self.face_rpn_cls_score_stride8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0, bias=True))

        self.face_rpn_cls_score_stride8_softmax = nn.Sequential(
            nn.Softmax(dim=1))

        self.face_rpn_bbox_pred_stride8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True))

        self.face_rpn_landmark_pred_stride8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=20, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.mobilenet0_conv0(x)
        x = self.mobilenet0_conv1(x)
        x = self.mobilenet0_conv2(x)
        x = self.mobilenet0_conv3(x)
        x = self.mobilenet0_conv4(x)
        x = self.mobilenet0_conv5(x)
        x = self.mobilenet0_conv6(x)
        x = self.mobilenet0_conv7(x)
        x = self.mobilenet0_conv8(x)
        x = self.mobilenet0_conv9(x)
        x10 = self.mobilenet0_conv10(x)
        x = self.mobilenet0_conv11(x10)
        x = self.mobilenet0_conv12(x)
        x = self.mobilenet0_conv13(x)
        x = self.mobilenet0_conv14(x)
        x = self.mobilenet0_conv15(x)
        x = self.mobilenet0_conv16(x)
        x = self.mobilenet0_conv17(x)
        x = self.mobilenet0_conv18(x)
        x = self.mobilenet0_conv19(x)
        x = self.mobilenet0_conv20(x)
        x = self.mobilenet0_conv21(x)
        x22 = self.mobilenet0_conv22(x)
        x = self.mobilenet0_conv23(x22)
        x = self.mobilenet0_conv24(x)
        x = self.mobilenet0_conv25(x)
        x = self.mobilenet0_conv26(x)
        # print(x.size())
        o1 = self.rf_c3_lateral(x) #256->64
        o2 = self.rf_c3_det_conv1(o1)
        o3 = self.rf_c3_det_context_conv1(o1)
        o4 = self.rf_c3_det_context_conv2(o3)
        o5 = self.rf_c3_det_context_conv3_1(o3)
        o6 = self.rf_c3_det_context_conv3_2(o5)
        o7 = torch.cat((o2, o4, o6), 1)
        o8 = self.rf_c3_det_concat_relu(o7)
        cls32 = self.face_rpn_cls_score_stride32(o8)
        cls32_shape = cls32.shape
        cls32 = torch.reshape(cls32, (batchsize, 2, -1, cls32_shape[3]))
        cls32 = self.face_rpn_cls_score_stride32_softmax(cls32)
        cls32 = torch.reshape(cls32, (batchsize, 4, -1, cls32_shape[3]))
        bbox32 = self.face_rpn_bbox_pred_stride32(o8)
        landmark32 = self.face_rpn_landmark_pred_stride32(o8)

        p1 = self.rf_c2_lateral(x22)
        p2 = self.rf_c3_upsampling(o1)
        p2 = F.adaptive_avg_pool2d(p2, (p1.shape[2], p1.shape[3]))

        p3 = p1 + p2
        p4 = self.rf_c2_aggr(p3)
        p5 = self.rf_c2_det_conv1(p4)
        p6 = self.rf_c2_det_context_conv1(p4)
        p7 = self.rf_c2_det_context_conv2(p6)
        p8 = self.rf_c2_det_context_conv3_1(p6)
        p9 = self.rf_c2_det_context_conv3_2(p8)
        p10 = torch.cat((p5, p7, p9), 1)
        p10 = self.rf_c2_det_concat_relu(p10)
        cls16 = self.face_rpn_cls_score_stride16(p10)
        cls16_shape = cls16.shape
        cls16 = torch.reshape(cls16, (batchsize, 2, -1, cls16_shape[3]))
        cls16 = self.face_rpn_cls_score_stride16_softmax(cls16)
        cls16 = torch.reshape(cls16, (batchsize, 4, -1, cls16_shape[3]))
        bbox16 = self.face_rpn_bbox_pred_stride16(p10)
        landmark16 = self.face_rpn_landmark_pred_stride16(p10)
        q1 = self.rf_c1_red_conv(x10)
        q2 = self.rf_c2_upsampling(p4)
        q2 = F.adaptive_avg_pool2d(q2, (q1.shape[2], q1.shape[3]))

        q3 = q1 + q2
        q4 = self.rf_c1_aggr(q3)
        q5 = self.rf_c1_det_conv1(q4)
        q6 = self.rf_c1_det_context_conv1(q4)
        q7 = self.rf_c1_det_context_conv2(q6)
        q8 = self.rf_c1_det_context_conv3_1(q6)
        q9 = self.rf_c1_det_context_conv3_2(q8)
        q10 = torch.cat((q5, q7, q9), 1)
        q10 = self.rf_c2_det_concat_relu(q10)
        cls8 = self.face_rpn_cls_score_stride8(q10)
        cls8_shape = cls8.shape
        cls8 = torch.reshape(cls8, (batchsize, 2, -1, cls8_shape[3]))
        cls8 = self.face_rpn_cls_score_stride8_softmax(cls8)
        cls8 = torch.reshape(cls8, (batchsize, 4, -1, cls8_shape[3]))
        bbox8 = self.face_rpn_bbox_pred_stride8(q10)
        landmark8 = self.face_rpn_landmark_pred_stride8(q10)

        detections = []
        detections.append(cls32)
        detections.append(bbox32)
        detections.append(landmark32)

        detections.append(cls16)
        detections.append(bbox16)
        detections.append(landmark16)

        detections.append(cls8)
        detections.append(bbox8)
        detections.append(landmark8)

        return detections


net = RetinaFace_MobileNet()
net.cuda()
from torchsummary import summary
summary(net, input_size=(3, 640, 640))
import pdb
pdb.set_trace()

img_dim = 640
input_size = (1, 3, img_dim, img_dim)
img = torch.FloatTensor(input_size[0], input_size[1], input_size[2], input_size[3])
net = add_flops_counting_methods(net)
net.start_flops_count()
feat = net(img)
faceboxes_flops = net.compute_average_flops_cost()
print('Net Flops:  {}'.format(flops_to_string(faceboxes_flops)))
print('Net Params: ' + get_model_parameters_number(net))


