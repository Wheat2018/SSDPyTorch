from .img_tranforms import *
from ....facedet.utils.bbox.centerface_target import CenterFaceTargetGenerator
import torch

target_generator = CenterFaceTargetGenerator(stages=3, stride=(8, 16, 32), valid_range=((16, 64), (64, 128), (128, 320)))

class preproc_centerface(object):

    def __init__(self, img_dim, rgb_means, ldmk_reg_type=None, **kwargs):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.ldmk_reg_type = ldmk_reg_type
        if self.ldmk_reg_type is not None:
            self.use_ldmk = True
        else:
            self.use_ldmk = False

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        # print(image.shape)


        if self.use_ldmk:
            boxes = targets[:, :4].copy()
            labels = targets[:, 4].copy()
            ldmks = targets[:, 5:].copy()
            # print('ldmks', ldmks.shape)
            # print('ldmks', ldmks)
            # import pdb
            # pdb.set_trace()
            image_t, boxes_t, labels_t, ldmks_t, pad_image_flag = crop_with_ldmk(image,
                                                                                 boxes,
                                                                                 labels,
                                                                                 ldmks,
                                                                                 self.img_dim)
            image_t = distort(image_t)
            image_t = pad_to_square(image_t, self.rgb_means, pad_image_flag)
            image_t, boxes_t, ldmks_t = mirror_with_ldmk(image_t, boxes_t, ldmks_t)
            height, width, _ = image_t.shape

            # image_tmp = image_t.copy()
            image_t = resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
            labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape
            # print("ldmks_t.shape after mirror", ldmks_t.shape)

            boxes_t[:, 0::2] /= (width / new_width)
            boxes_t[:, 1::2] /= (height / new_height)

            mask_landmarks = (ldmks_t == -1).all(axis=1)
            ldmks_t[:, 0::3] /= (width / new_width)
            ldmks_t[:, 1::3] /= (height / new_height)
            ldmks_t[mask_landmarks] = -1.0
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            ldmks_t = torch.from_numpy(ldmks_t.astype(np.float32))
            # if ldmks_t.max()>640:
            #     import pdb
            #     pdb.set_trace()
            #     print(ldmks_t.max())
            # print("ldmk", ldmks.size())
            if self.ldmk_reg_type == 'coord':
                box_target, cls_target, ctr_target, box_mask, ldmk_target, ldmk_mask = \
                target_generator.generate_targets(image_t, boxes, ldmks_t, self.ldmk_reg_type)
                return image_t, box_target, cls_target, ctr_target, box_mask, ldmk_target, ldmk_mask
            elif self.ldmk_reg_type == 'heatmap':
                box_target, cls_target, ctr_target, box_mask, ldmk_target = \
                target_generator.generate_targets(image_t, boxes, ldmks_t, self.ldmk_reg_type)

                # import pdb
                # import cv2
                # # tmp_ldmk_target = ldmk_target[:6400, :].view(80, 80, 5).permute(2, 0, 1).numpy()
                # tmp_ldmk_target = ldmk_target[:16384, :].view(128, 128, 5).permute(2, 0, 1).numpy()
                # for i in range(0, 5):
                #     heatmap = cv2.applyColorMap(np.uint8(255 * tmp_ldmk_target[i]), cv2.COLORMAP_JET)
                #     cv2.imwrite('heatmap_' + str(i) + '.jpg', heatmap)
                # feat = (tmp_ldmk_target[0] + tmp_ldmk_target[1] + tmp_ldmk_target[2] + tmp_ldmk_target[3] + tmp_ldmk_target[4])
                # heatmap_all = cv2.applyColorMap(np.uint8(255 * feat), cv2.COLORMAP_JET)
                # threshold = 0.1
                # heatmap_all[np.where(feat < threshold)] = 0
                #
                # resized = cv2.resize(image_tmp, (320, 320), interpolation=cv2.INTER_AREA)
                # heatmap_all = cv2.resize(heatmap_all, (320, 320), interpolation=cv2.INTER_AREA)
                # cv2.imwrite('heatmap_all.jpg', heatmap_all*0.5 + resized)
                # pdb.set_trace()

                # import pdb
                # import cv2
                # pdb.set_trace()
                # # tmp_ldmk_target = ldmk_target[:6400, :].view(80, 80, 5).permute(2, 0, 1).numpy()
                # tmp_cls_target = cls_target[:16384, :].view(128, 128, 1).permute(2, 0, 1).numpy()
                # for i in range(0, 1):
                #     heatmap = cv2.applyColorMap(np.uint8(255 * tmp_cls_target[i]), cv2.COLORMAP_JET)
                #     cv2.imwrite('heatmap_' + str(i) + '.jpg', heatmap)
                # feat = (tmp_cls_target[0])
                # heatmap_all = cv2.applyColorMap(np.uint8(255 * feat), cv2.COLORMAP_JET)
                # threshold = 0.1
                # heatmap_all[np.where(feat < threshold)] = 0
                #
                # resized = cv2.resize(image_tmp, (320, 320), interpolation=cv2.INTER_AREA)
                # heatmap_all = cv2.resize(heatmap_all, (320, 320), interpolation=cv2.INTER_AREA)
                # cv2.imwrite('heatmap_all.jpg', heatmap_all*0.5 + resized)
                # pdb.set_trace()


                return image_t, box_target, cls_target, ctr_target, box_mask, ldmk_target
        else:
            boxes = targets[:, :-1].copy()
            labels = targets[:, -1].copy()

            image_t, boxes_t, labels_t, pad_image_flag = crop(image, boxes, labels, self.img_dim)
            image_t = distort(image_t)
            image_t = pad_to_square(image_t,self.rgb_means, pad_image_flag)
            image_t, boxes_t = mirror(image_t, boxes_t)
            height, width, _ = image_t.shape
            image_t = resize_subtract_mean(image_t, self.img_dim, self.rgb_means)

            labels_t = np.expand_dims(labels_t, 1)
            _, new_height, new_width = image_t.shape
            boxes_t[:, 0::2] /= (width / new_width)
            boxes_t[:, 1::2] /= (height / new_height)
            boxes = np.hstack((boxes_t, labels_t))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            box_target, cls_target, ctr_target, reg_mask = \
            target_generator.generate_targets(image_t, boxes)
            return image_t, box_target, cls_target, ctr_target, reg_mask





