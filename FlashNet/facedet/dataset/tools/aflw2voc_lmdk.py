#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, h5py, cv2, sys, shutil
import numpy as np
from xml.dom.minidom import Document
import pdb

rootdir = "/mnt/lustre/geyongtao/dataset/AFLW"
convet2yoloformat = False
convert2vocformat = True
# resized_dim = (48, 48)

# 最小取8大小的脸，并且补齐
minsize2select = 8
usepadding = True

datasetprefix = "./wider_face"  #


def convertimgset(img_set="train"):
    typical_pose, atypical_pose = 0, 0
    imgdir = os.path.join(rootdir, "JPEGImages")
    gt_filepath = os.path.join(rootdir, "train_gt.txt")
    imagesdir = rootdir + "/JPEGImages"
    vocannotationdir = rootdir + "/Annotations"
    labelsdir = rootdir + "/labels"
    if not os.path.exists(imagesdir):
        os.makedirs(imagesdir)
    if not os.path.exists(rootdir + "/ImageSets"):
        os.makedirs(rootdir + "/ImageSets")
    if not os.path.exists(rootdir + "/ImageSets/Main"):
        os.makedirs(rootdir + "/ImageSets/Main")

    if convet2yoloformat:
        if not os.path.exists(labelsdir):
            os.mkdir(labelsdir)
    if convert2vocformat:
        if not os.path.exists(vocannotationdir):
            os.mkdir(vocannotationdir)
    index = 0

    with open(gt_filepath, 'r') as gt_file:
        gt_file.readline()
        while (True):  # and len(faces)<10
            # import pdb
            # pdb.set_trace()
            line = gt_file.readline().split()
            landmark_line = line[5:]
            box_line = line[1:5]
            filename = line[0]
            imgpath = os.path.join(imgdir, filename)
            img = cv2.imread(imgpath)
            bboxes = []
            bboxes_landmark = []

            xmin = int(float(box_line[0]))
            ymin = int(float(box_line[1]))
            xmax = int(float(box_line[2]))
            ymax = int(float(box_line[3]))
            blur = -1
            expression = -1
            illumination = -1
            invalid = -1
            occlusion = -1
            pose = -1

            x1 = float(landmark_line[0])
            y1 = float(landmark_line[1])
            visible1 = float(1.0)
            x2 = float(landmark_line[2])
            y2 = float(landmark_line[3])
            visible2 = float(1.0)
            x3 = float(landmark_line[4])
            y3 = float(landmark_line[5])
            visible3 = float(1.0)
            x4 = float(landmark_line[6])
            y4 = float(landmark_line[7])
            visible4 = float(1.0)
            x5 = float(landmark_line[8])
            y5 = float(landmark_line[9])
            visible5 = float(1.0)
            blur_score = -1

            width = xmax-xmin+1
            height = ymax-ymin+1

            bbox = (xmin, ymin, xmax, ymax, blur, expression, illumination, invalid, occlusion, pose)
            # visible -1 表示未标注 0表示不可见 1表示可见
            bbox_landmark = (x1, y1, visible1, x2, y2, visible2, x3, y3, visible3, x4, y4, visible4, x5, y5, visible5, blur_score)
            # import pdb
            # pdb.set_trace()
            # face=img[x:x2,y:y2]
            if width >= minsize2select and height >= minsize2select:
                bboxes.append(bbox)
                bboxes_landmark.append(bbox_landmark)
                # cv2.rectangle(showimg, (xmin, ymin), (x2, y2), (0, 255, 0))
            # else:
                # saveimg[ymin:ymax, xmin:xmax, :] = (104, 117, 123)
                # cv2.rectangle(showimg, (xmin, ymin), (x2, y2), (0, 0, 255))


            # filename = filename.replace("/", "_")
            path, _ = os.path.split(imagesdir + "/" + filename)  # 返回一个路径名和文件名
            if not os.path.exists(path):
                os.makedirs(path)
            if len(bboxes) == 0:
                print("no face:", filename)
                continue
            # cv2.imwrite(imagesdir + "/" + box_filename, saveimg)
            imgfilepath = filename[:-4]
            with open(rootdir + "/ImageSets/Main/" + img_set + ".txt", 'a') as f_set:
                f_set.write(imgfilepath + '\n')
            if convert2vocformat:
                xmlpath = vocannotationdir + "/" + filename
                path, _ = os.path.split(xmlpath)
                if not os.path.exists(path):
                    os.makedirs(path)

                xmlpath = xmlpath[:-3] + "xml"
                doc = Document()
                annotation = doc.createElement('annotation')
                doc.appendChild(annotation)
                folder = doc.createElement('folder')
                folder_name = doc.createTextNode('widerface')
                folder.appendChild(folder_name)
                annotation.appendChild(folder)
                filenamenode = doc.createElement('filename')
                filename_name = doc.createTextNode(filename)
                filenamenode.appendChild(filename_name)
                annotation.appendChild(filenamenode)
                source = doc.createElement('source')
                annotation.appendChild(source)
                database = doc.createElement('database')
                database.appendChild(doc.createTextNode('WIDER Database'))
                source.appendChild(database)
                annotation_s = doc.createElement('annotation')
                annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
                source.appendChild(annotation_s)
                image = doc.createElement('image')
                image.appendChild(doc.createTextNode('flickr'))
                source.appendChild(image)
                flickrid = doc.createElement('flickrid')
                flickrid.appendChild(doc.createTextNode('-1'))
                source.appendChild(flickrid)
                owner = doc.createElement('owner')
                annotation.appendChild(owner)
                flickrid_o = doc.createElement('flickrid')
                flickrid_o.appendChild(doc.createTextNode('gyt'))
                owner.appendChild(flickrid_o)
                name_o = doc.createElement('name')
                name_o.appendChild(doc.createTextNode('gyt'))
                owner.appendChild(name_o)
                size = doc.createElement('size')
                annotation.appendChild(size)
                width = doc.createElement('width')
                width.appendChild(doc.createTextNode(str(img.shape[1])))
                height = doc.createElement('height')
                height.appendChild(doc.createTextNode(str(img.shape[0])))
                depth = doc.createElement('depth')
                depth.appendChild(doc.createTextNode(str(img.shape[2])))
                size.appendChild(width)
                size.appendChild(height)
                size.appendChild(depth)
                segmented = doc.createElement('segmented')
                segmented.appendChild(doc.createTextNode('0'))
                annotation.appendChild(segmented)
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    bbox_landmark = bboxes_landmark[i]

                    objects = doc.createElement('object')
                    annotation.appendChild(objects)
                    object_name = doc.createElement('name')
                    object_name.appendChild(doc.createTextNode('face'))
                    objects.appendChild(object_name)

                    pose = doc.createElement('pose')
                    # pose.appendChild(doc.createTextNode('Unspecified'))
                    pose.appendChild(doc.createTextNode(str(bbox[-1])))

                    if bbox[-1] == 1:
                        atypical_pose += 1
                    if bbox[-1] == 0:
                        typical_pose += 1
                    # print('typical_pose', typical_pose)
                    # print('atypical_pose', atypical_pose)
                    objects.appendChild(pose)

                    truncated = doc.createElement('truncated')
                    truncated.appendChild(doc.createTextNode('1'))
                    objects.appendChild(truncated)

                    difficult = doc.createElement('difficult')
                    difficult.appendChild(doc.createTextNode('0'))
                    objects.appendChild(difficult)

                    blur = doc.createElement('blur')
                    blur.appendChild(doc.createTextNode(str(bbox[4])))
                    objects.appendChild(blur)

                    bndbox = doc.createElement('bndbox')
                    objects.appendChild(bndbox)
                    xmin = doc.createElement('xmin')
                    xmin.appendChild(doc.createTextNode(str(bbox[0])))
                    bndbox.appendChild(xmin)
                    ymin = doc.createElement('ymin')
                    ymin.appendChild(doc.createTextNode(str(bbox[1])))
                    bndbox.appendChild(ymin)
                    xmax = doc.createElement('xmax')
                    xmax.appendChild(doc.createTextNode(str(bbox[2])))
                    bndbox.appendChild(xmax)
                    ymax = doc.createElement('ymax')
                    ymax.appendChild(doc.createTextNode(str(bbox[3])))
                    bndbox.appendChild(ymax)

                    x1 = doc.createElement('x1')
                    x1.appendChild(doc.createTextNode(str(bbox_landmark[0])))
                    bndbox.appendChild(x1)
                    y1 = doc.createElement('y1')
                    y1.appendChild(doc.createTextNode(str(bbox_landmark[1])))
                    bndbox.appendChild(y1)
                    v1 = doc.createElement('v1')
                    v1.appendChild(doc.createTextNode(str(bbox_landmark[2])))
                    bndbox.appendChild(v1)

                    x2 = doc.createElement('x2')
                    x2.appendChild(doc.createTextNode(str(bbox_landmark[3])))
                    bndbox.appendChild(x2)
                    y2 = doc.createElement('y2')
                    y2.appendChild(doc.createTextNode(str(bbox_landmark[4])))
                    bndbox.appendChild(y2)
                    v2 = doc.createElement('v2')
                    v2.appendChild(doc.createTextNode(str(bbox_landmark[5])))
                    bndbox.appendChild(v2)

                    x3 = doc.createElement('x3')
                    x3.appendChild(doc.createTextNode(str(bbox_landmark[6])))
                    bndbox.appendChild(x3)
                    y3 = doc.createElement('y3')
                    y3.appendChild(doc.createTextNode(str(bbox_landmark[7])))
                    bndbox.appendChild(y3)
                    v3 = doc.createElement('v3')
                    v3.appendChild(doc.createTextNode(str(bbox_landmark[8])))
                    bndbox.appendChild(v3)

                    x4 = doc.createElement('x4')
                    x4.appendChild(doc.createTextNode(str(bbox_landmark[9])))
                    bndbox.appendChild(x4)
                    y4 = doc.createElement('y4')
                    y4.appendChild(doc.createTextNode(str(bbox_landmark[10])))
                    bndbox.appendChild(y4)
                    v4 = doc.createElement('v4')
                    v4.appendChild(doc.createTextNode(str(bbox_landmark[11])))
                    bndbox.appendChild(v4)

                    x5 = doc.createElement('x5')
                    x5.appendChild(doc.createTextNode(str(bbox_landmark[12])))
                    bndbox.appendChild(x5)
                    y5 = doc.createElement('y5')
                    y5.appendChild(doc.createTextNode(str(bbox_landmark[13])))
                    bndbox.appendChild(y5)
                    v5 = doc.createElement('v5')
                    v5.appendChild(doc.createTextNode(str(bbox_landmark[14])))
                    bndbox.appendChild(v5)
                    blur_score = doc.createElement('blur_score')
                    blur_score.appendChild(doc.createTextNode(str(bbox_landmark[15])))
                    bndbox.appendChild(blur_score)

                f = open(xmlpath, "w")
                f.write(doc.toprettyxml(indent=''))
                f.close()
            index = index + 1


def generatetxt(img_set="train"):
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"
    f = open(rootdir + "/" + img_set + ".txt", "w")
    with open(gtfilepath, 'r') as gtfile:
        while (True):  # and len(faces)<10
            filename = gtfile.readline()[:-1]
            if (filename == ""):
                break;
            # filename = filename.replace("/", "_")
            # imgfilepath = datasetprefix + "/images/" + filename
            imgfilepath = filename
            f.write(imgfilepath + '\n')
            numbbox = int(gtfile.readline())
            for i in range(numbbox):
                line = gtfile.readline()
    f.close()


def generatevocsets(img_set="train"):
    if not os.path.exists(rootdir + "/ImageSets"):
        os.mkdir(rootdir + "/ImageSets")
    if not os.path.exists(rootdir + "/ImageSets/Main"):
        os.mkdir(rootdir + "/ImageSets/Main")
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"
    f = open(rootdir + "/ImageSets/Main/" + img_set + ".txt", 'w')
    with open(gtfilepath, 'r') as gtfile:
        while (True):  # and len(faces)<10
            filename = gtfile.readline()[:-1]
            if (filename == ""):
                break;
            filename = filename.replace("/", "_")
            imgfilepath = filename[:-4]
            f.write(imgfilepath + '\n')
            numbbox = int(gtfile.readline())
            for i in range(numbbox):
                line = gtfile.readline()
    f.close()


def convertdataset():
    # img_sets = ["train", "val"]
    img_sets = ["train"]
    # img_sets = ["val"]
    for img_set in img_sets:
        convertimgset(img_set)
        # generatetxt(img_set)
        # generatevocsets(img_set)


if __name__ == "__main__":
    convertdataset()
    # generatetxt('val')
    # shutil.move(rootdir + "/" + "train.txt", rootdir + "/" + "trainval.txt")
    # shutil.move(rootdir + "/" + "val.txt", rootdir + "/" + "test.txt")
    shutil.move(rootdir + "/ImageSets/Main/" + "train.txt", rootdir + "/ImageSets/Main/" + "trainval.txt")
    # shutil.move(rootdir + "/ImageSets/Main/" + "val.txt", rootdir + "/ImageSets/Main/" + "test.txt")
