import cv2
import numpy as np
import random
from ....facedet.utils.bbox.box_utils import matrix_iof

def crop(image, boxes, labels, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
#        mask_b = np.minimum(b_w_t, b_h_t) > 16.0
        mask_b = np.minimum(b_w_t, b_h_t) > 5.0
        boxes_t = boxes_t[mask_b]

        # b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        # b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        # mask_small_face = np.minimum(b_w_t, b_h_t) < 24.0
        # boxes_tt = boxes_t[mask_small_face]
        #
        # small_b_w = (boxes_tt[:, 2] - boxes_tt[:, 0])
        # small_b_h = (boxes_tt[:, 3] - boxes_tt[:, 1])
        # boxes_t[mask_small_face][:,0] = boxes_tt[:,0] - small_b_w / 2
        # boxes_t[mask_small_face][:,2] = boxes_tt[:,2] + small_b_w / 2
        # boxes_t[mask_small_face][:,1] = boxes_tt[:,1] - small_b_h / 2
        # boxes_t[mask_small_face][:,3] = boxes_tt[:,3] + small_b_h / 2


        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag

def crop_with_ldmk(image, boxes, labels, ldmks, img_dim):
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        if random.uniform(0, 1) <= 0.2:
            scale = 1
        else:
            scale = random.uniform(0.3, 1.)
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            l = 0
        else:
            l = random.randrange(width - w)
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()
        ldmks_t = ldmks[mask_a].copy()
        # print('mask_a',mask_a)

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        # print('landmark before',ldmks_t)
        # print('tile',np.tile(roi[:2], 5))
        mask_ldmks = (ldmks_t!=-1).all(axis=1)

        # print('mask_ldmks',mask_ldmks)
        # print(ldmks_t[mask_ldmks].shape)
        # print(ldmks_t[mask_ldmks])
        # (roi[0], roi[1], 0)) -> (left, top, 0)
        if ldmks_t.shape[1]==15:
            ldmks_t[mask_ldmks] = ldmks_t[mask_ldmks] - np.tile(np.array((roi[0], roi[1], 0)), 5)
        else:
            ldmks_t[mask_ldmks] = ldmks_t[mask_ldmks] - np.tile(roi[:2], 5)
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

	    # make sure that the cropped image contains at least one face > 5 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        # mask_b = np.minimum(b_w_t, b_h_t) > 16.0
        mask_b = np.minimum(b_w_t, b_h_t) > 5.0
        boxes_t = boxes_t[mask_b]
        ldmks_t = ldmks_t[mask_b]
        # mask small face's landmark
        # b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        # b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        # mask__sface_ldmk = np.minimum(b_w_t, b_h_t) < 20.0
        # ldmks_t[mask__sface_ldmk] = -1.0

        # b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        # b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        # mask_small_face = np.minimum(b_w_t, b_h_t) < 24.0
        # boxes_tt = boxes_t[mask_small_face]
        #
        # small_b_w = (boxes_tt[:, 2] - boxes_tt[:, 0])
        # small_b_h = (boxes_tt[:, 3] - boxes_tt[:, 1])
        # boxes_t[mask_small_face][:,0] = boxes_tt[:,0] - small_b_w / 2
        # boxes_t[mask_small_face][:,2] = boxes_tt[:,2] + small_b_w / 2
        # boxes_t[mask_small_face][:,1] = boxes_tt[:,1] - small_b_h / 2
        # boxes_t[mask_small_face][:,3] = boxes_tt[:,3] + small_b_h / 2
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False
        # print('landmark after crop', ldmks_t.shape)
        return image_t, boxes_t, labels_t, ldmks_t, pad_image_flag
    # print('beyond 250 times')

    return image, boxes, labels, ldmks, pad_image_flag

def distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def expand(image, boxes, fill, p):
    if random.randrange(2):
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p)
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width)
    top = random.randint(0, h - height)

    boxes_t = boxes.copy()
    boxes_t[:, :2] += (left, top)
    boxes_t[:, 2:] += (left, top)
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    expand_image[top:top + height, left:left + width] = image
    image = expand_image

    return image, boxes_t


def mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def draw_results(img_origin, bboxes_scores, ldmks, img_save_name):
    """
    Save predicted results, including bbox and score into text file.
    Args:
        image_path (string): file name.
        bboxes_scores (np.array|list): the predicted bboxed and scores, layout
            is (xmin, ymin, xmax, ymax, score)
        output_dir (string): output directory.
    """
    # import pdb
    # pdb.set_trace()
    for box_score in bboxes_scores:
        xmin, ymin, xmax, ymax = box_score
        score = 1.0
        if score > 0.4:
            if img_save_name=="after_mirror.jpg":
                cv2.rectangle(img_origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
            cv2.rectangle(img_origin, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
            # cv2.putText(img_origin, str('%0.2f' % score), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    for idx, landmark in enumerate(ldmks):
        landmark = landmark.reshape(5, 2)
        if landmark.max() == -1.0:
            score = 0.0
        else:
            score = 1.0
        if score > 0.4:
            for point in landmark:
                # import pdb
                # pdb.set_trace()
                # print('point',point)
                if img_save_name == "after_mirror.jpg":
                    cv2.circle(img_origin, (int(point[0]), int(point[1])), radius=1, color=(0, 255, 255), thickness=1)
                cv2.circle(img_origin, (int(point[0]), int(point[1])), radius=1, color=(0, 0, 255), thickness=1)


    cv2.imwrite(img_save_name, img_origin)


def mirror_with_ldmk(image, boxes, landmarks):
    _, width, _ = image.shape
    # import pdb
    # pdb.set_trace()
    # draw_results(image, boxes, landmarks, img_save_name="before_mirror.jpg")
    # print('image.shape', image.shape)

    if random.randrange(2):
        # print('before',image[:,:,0])
        image = image[:, ::-1].copy()
        # print('after',image[:,:,0])
        boxes = boxes.copy()
        landmarks = landmarks.copy()
        # print('boxes', boxes)
        # print('boxes[:, 0::2]', boxes[:, 0::2])
        # print('boxes[:, 2::-2]', boxes[:, 2::-2])
        boxes[:, 0::2] = width - 1 - boxes[:, 2::-2]
        # boxes[:, 0::2] = width - 1 - boxes[:, 2::-2]
        mask_landmarks = (landmarks==-1).all(axis=1)
        # print('mask_landmarks.shape', mask_landmarks.shape)
        landmarks[:, 0::3] = width - 1 - landmarks[:, 0::3]
        # landmarks[:, 0::2] = width - landmarks[:, 0::2]
        if landmarks.shape[1]==10:
            flip_order = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7]
        elif landmarks.shape[1]==15:
            flip_order = [3, 4, 5, 0, 1, 2, 6, 7, 8, 12, 13, 14, 9, 10, 11]
        else:
            raise NotImplementedError
        # print('landmarks before flip', landmarks)
        landmarks = landmarks[:, flip_order]
        # print('landmarks after flip', landmarks)
        landmarks[mask_landmarks] = -1

        # draw_results(image, boxes, landmarks, img_save_name="after_mirror.jpg")
        # import pdb
        # pdb.set_trace()
        # if mask_landmarks.shape[0] != 0:
            # print('width', width)
            # print('landmarks', landmarks)
            # print('landmarks[:, 0::2] before', landmarks[:, 0::2])
            # print('landmarks[:, 8::-2]', landmarks[:, 8::-2])
            # landmarks[:, 0::2] = width - 1 - landmarks[:, 0::2]
            # landmarks[mask_landmarks] = -1
            # print('landmarks[:, 0::2] after', landmarks[:, 0::2])
            # print('landmarks after', landmarks)
        # print(landmarks)
    return image, boxes, landmarks

def pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = rgb_mean
    image_t[0:0 + height, 0:0 + width] = image
    return image_t

def resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)

def rotate(image, boxes, labels, ldmks):

    ################
    #rect = cv2.minAreaRect(cnt) #最小外接矩形
    #rect = cv2.boundingRect(cnt) #正外接矩形
    ##################################
    height, width, _ = image.shape
    img_center = (int(width / 2), int(height / 2))
    num_box = boxes.shape[0]
    # print('boxes',boxes.shape)
    # print('ldmks', ldmks[0])
    # print('image.shape', image.shape)
    if boxes.min() > 64:
        rotate_angle = np.random.randint(-20, 20)
        # print('rotate_angle',rotate_angle)
        affine_mat = cv2.getRotationMatrix2D(img_center, rotate_angle, 1)
        image_rotate = cv2.warpAffine(image, affine_mat, (width, height))
        print('image_rotate.shape', image_rotate.shape)
        for i in range(num_box):
            box_w = boxes[i][2] - boxes[i][0]
            box_h = boxes[i][3] - boxes[i][1]
            box_w = np.sqrt(box_w * box_h)
            box_h = box_w
            x_c = int((boxes[i][0] + boxes[i][2]) / 2)
            y_c = int((boxes[i][1] + boxes[i][3]) / 2)
            points = np.array([[x_c, y_c]])
            ones = np.ones(shape=(len(points), 1))
            points_ones = np.hstack([points, ones])
            # print('points_ones', points_ones)
            # transform points
            transformed_points = affine_mat.dot(points_ones.T).T
            # cv2.ellipse(img, (256, 256), (150, 100), 0, 0, 180, 250, -1)
            x_min = int(transformed_points[0][0] - box_w / 2)
            y_min = int(transformed_points[0][1] - box_h / 2)
            x_max = int(transformed_points[0][0] + box_w / 2)
            y_max = int(transformed_points[0][1] + box_h / 2)
            # print('before rotate boxes[i]', boxes[i])
            boxes[i] = np.array([x_min, y_min, x_max,  y_max])
            # print('after rotate boxes[i]', boxes[i])
            # print('ldmks[i]!=-1).all(axis=0)', (ldmks[i]!=-1).all(axis=0))
            # print('before ldmks[i]', ldmks[i])
            if (ldmks[i]!=-1).all(axis=0):
                ldmks_one_face = ldmks[i].reshape(5, 2)
                ones = np.ones(shape=(len(ldmks_one_face), 1))
                ldmks_one_face = np.hstack([ldmks_one_face, ones])
                transformed_ldmks_one_face = affine_mat.dot(ldmks_one_face.T).T
                ldmks[i] = transformed_ldmks_one_face.reshape(1, 10)[0]
            # if (ldmks[i]!=-1).all(axis=0) and i == 0:
                # print('ldmks_one_face', ldmks_one_face)
                # print('transformed_ldmks_one_face', transformed_ldmks_one_face)
                # print('after ldmks[0]', ldmks)
        image_rotate = image_rotate.copy()
        # print(ldmks)
        return image_rotate, boxes, labels, ldmks
    else:
        return image, boxes, labels, ldmks