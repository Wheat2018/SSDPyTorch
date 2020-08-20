from .img_tranforms import *
import torch



class preproc_kpface(object):

    def __init__(self, img_dim, rgb_means, target_generator=None, **kwargs):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.target_generator = target_generator

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        # print(image.shape)
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ldmks = targets[:, 5:].copy()
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

        box_target, box_mask, ldmk_target, hard_face_target = \
        self.target_generator.generate_targets(image_t, boxes, ldmks_t)

        return image_t, box_target, box_mask, ldmk_target, hard_face_target


class preproc_kpface_vis(object):

    def __init__(self, img_dim, rgb_means, target_generator=None, **kwargs):
        self.img_dim = img_dim
        self.rgb_means = rgb_means
        self.target_generator = target_generator

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt"
        # print(image.shape)
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        ldmks = targets[:, 5:].copy()
        image_t, boxes_t, labels_t, ldmks_t, pad_image_flag = crop_with_ldmk(image,
                                                                             boxes,
                                                                             labels,
                                                                             ldmks,
                                                                             self.img_dim)
        # image_t = distort(image_t)
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

        box_target, box_mask, ldmk_target, hard_face_target = \
        self.target_generator.generate_targets(image_t, boxes, ldmks_t)

        return image_t, box_target, box_mask, ldmk_target, hard_face_target
