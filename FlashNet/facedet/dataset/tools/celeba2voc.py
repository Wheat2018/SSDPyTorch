import cv2,h5py,os
import numpy as np
from xml.dom.minidom import Document
import progressbar
rootdir="../"
imgdir=rootdir+"Img/img_celeba"

landmarkpath=rootdir+"Anno/list_landmarks_celeba.txt"
bboxpath=rootdir+"Anno/list_bbox_celeba.txt"
vocannotationdir=rootdir+"/"+"Annotations"
labelsdir=rootdir+"/"+"labels"

convet2yoloformat=True
convert2vocformat=True

resized_dim=(48, 48)

datasetprefix="/home/yanhe/data/CelebA/images/"
progress = progressbar.ProgressBar(widgets=[
    progressbar.Percentage(),
    ' (', progressbar.SimpleProgress(), ') ',
    ' (', progressbar.Timer(), ') ',
    ' (', progressbar.ETA(), ') ',])
def drawbboxandlandmarks(img,bbox,landmark):
    cv2.rectangle(img,(bbox[0], bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0, 255, 0))
    for i in range(int(len(landmark)/2)):
        cv2.circle(img,(int(landmark[2*i]), int(landmark[2*i+1])), 2, (0, 0, 255))

def loadgt():
    imgpaths=[]
    landmarks=[]
    bboxes=[]
    with open(landmarkpath) as landmarkfile:
        lines=landmarkfile.readlines()
        lines=lines[2:]
        for line in lines:
            landmarkline=line.split()
            imgpath=landmarkline[0]
            imgpaths.append(imgpath)
            landmarkline=landmarkline[1:]
            landmark=[int(str) for str in landmarkline]
            landmarks.append(landmark)
    with open(bboxpath) as bboxfile:
        lines=bboxfile.readlines()
        lines=lines[2:]
        for line in lines:
            bboxline=line.split()
            imgpath=bboxline[0]
            bboxline=bboxline[1:]
            bbox=[int(bb) for bb in bboxline]
            bboxes.append(bbox)
    return imgpaths,bboxes,landmarks

def generate_hdf5():
    imgpaths,bboxes,landmarks=loadgt()
    numofimg=len(imgpaths)
    faces=[]
    labels=[]
    #numofimg=2
    for i in range(numofimg):
        imgpath=imgdir+"/"+imgpaths[i]
        print(i)#,imgpath)
        bbox=bboxes[i]
        landmark=landmarks[i]
        img=cv2.imread(imgpath)
        if bbox[2]<=0 or bbox[3]<=0:
            continue
        face=img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
        face=cv2.resize(face,resized_dim)
        faces.append(face)
        label=[]
        label.append(1)
        for i in range(len(bbox)):
            label.append(bbox[i])
        for i in range(len(landmark)):
            lm=landmark[i]
            if i%2==0:
                lm=(lm-bbox[0])*1.0/(bbox[2])
            else:
                lm=(lm-bbox[1])*1.0/(bbox[3])
            label.append(lm)
        labels.append(label)
    faces=np.asarray(faces)
    labels=np.asarray(labels)
    f=h5py.File('train.h5','w')
    f['data']=faces.astype(np.float32)
    f['labels']=labels.astype(np.float32)
    f.close()
def viewginhdf5():
    f = h5py.File('train.h5','r')
    f.keys()
    faces=f['data'][:]
    labels=f['labels'][:]
    for i in range(len(faces)):
        print(i)
        face=faces[i].astype(np.uint8)
        label=labels[i]
        bbox=label[1:4]
        landmark=label[5:]
        for i in range(int(len(landmark)/2)):
            cv2.circle(face,(int(landmark[2*i]*resized_dim[0]),int(landmark[2*i+1]*resized_dim[1])),1,(0,0,255))
        cv2.imshow("img",face)
        cv2.waitKey()
    f.close()
def showgt():
    landmarkfile=open(landmarkpath)
    bboxfile=open(bboxpath)
    numofimgs=int(landmarkfile.readline())
    _=landmarkfile.readline()
    _=bboxfile.readline()
    _=bboxfile.readline()
    index=0
    pbar = progress.start()
    if convet2yoloformat:
        if not os.path.exists(labelsdir):
            os.mkdir(labelsdir)
    if convert2vocformat:
        if not os.path.exists(vocannotationdir):
            os.mkdir(vocannotationdir)
#    while(index<numofimgs):
    for i in pbar(range(numofimgs)):
        #pbar.update(int((index/(numofimgs-1))*10000))
        landmarkline=landmarkfile.readline().split()
        filename=landmarkline[0]
        #sys.stdout.write("\r"+str(index)+":"+filename)
        #sys.stdout.flush()
        imgpath=imgdir+"/"+filename
        img=cv2.imread(imgpath)
        landmarkline=landmarkline[1:]
        landmark=[int(pt) for pt in landmarkline]
        bboxline=bboxfile.readline().split()
        imgpath2=imgdir+"/"+bboxline[0]
        bboxline=bboxline[1:]
        bbox=[int(bb) for bb in bboxline]
        drawbboxandlandmarks(img,bbox,landmark)
        if convet2yoloformat:
            height=img.shape[0]
            width=img.shape[1]
            txtpath=labelsdir+"/"+filename
            txtpath=txtpath[:-3]+"txt"
            ftxt=open(txtpath,'w')
            xcenter=(bbox[0]+bbox[2]*0.5)/width
            ycenter=(bbox[1]+bbox[3]*0.5)/height
            wr=bbox[2]*1.0/width
            hr=bbox[3]*1.0/height
            line="0 "+str(xcenter)+" "+str(ycenter)+" "+str(wr)+" "+str(hr)+"\n"
            ftxt.write(line)
            ftxt.close()
        if convert2vocformat:
            xmlpath=vocannotationdir+"/"+filename
            xmlpath=xmlpath[:-3]+"xml"
            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)
            folder = doc.createElement('folder')
            folder_name = doc.createTextNode('CelebA')
            folder.appendChild(folder_name)
            annotation.appendChild(folder)
            filenamenode = doc.createElement('filename')
            filename_name = doc.createTextNode(filename)
            filenamenode.appendChild(filename_name)
            annotation.appendChild(filenamenode)
            source = doc.createElement('source')
            annotation.appendChild(source)
            database = doc.createElement('database')
            database.appendChild(doc.createTextNode('CelebA Database'))
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
            flickrid_o.appendChild(doc.createTextNode('tdr'))
            owner.appendChild(flickrid_o)
            name_o = doc.createElement('name')
            name_o.appendChild(doc.createTextNode('yanyu'))
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
            for i in range(1):
                objects = doc.createElement('object')
                annotation.appendChild(objects)
                object_name = doc.createElement('name')
                object_name.appendChild(doc.createTextNode('face'))
                objects.appendChild(object_name)
                pose = doc.createElement('pose')
                pose.appendChild(doc.createTextNode('Unspecified'))
                objects.appendChild(pose)
                truncated = doc.createElement('truncated')
                truncated.appendChild(doc.createTextNode('1'))
                objects.appendChild(truncated)
                difficult = doc.createElement('difficult')
                difficult.appendChild(doc.createTextNode('0'))
                objects.appendChild(difficult)
                bndbox = doc.createElement('bndbox')
                objects.appendChild(bndbox)
                xmin = doc.createElement('xmin')
                xmin.appendChild(doc.createTextNode(str(bbox[0])))
                bndbox.appendChild(xmin)
                ymin = doc.createElement('ymin')
                ymin.appendChild(doc.createTextNode(str(bbox[1])))
                bndbox.appendChild(ymin)
                xmax = doc.createElement('xmax')
                xmax.appendChild(doc.createTextNode(str(bbox[0]+bbox[2])))
                bndbox.appendChild(xmax)
                ymax = doc.createElement('ymax')
                ymax.appendChild(doc.createTextNode(str(bbox[1]+bbox[3])))
                bndbox.appendChild(ymax)
            f=open(xmlpath,"w")
            f.write(doc.toprettyxml(indent = ''))
            f.close()
        cv2.imshow("img",img)
        cv2.waitKey(1)
        index=index+1
    pbar.finish()

def generatetxt(trainratio=0.7,valratio=0.2,testratio=0.1):
    files=os.listdir(labelsdir)
    ftrain=open(rootdir+"/"+"train.txt","w")
    fval=open(rootdir+"/"+"val.txt","w")
    ftrainval=open(rootdir+"/"+"trainval.txt","w")
    ftest=open(rootdir+"/"+"test.txt","w")
    index=0
    for i in range(len(files)):
        filename=files[i]
        filename=datasetprefix+filename[:-3]+"jpg"+"\n"
        if i<trainratio*len(files):
            ftrain.write(filename)
            ftrainval.write(filename)
        elif i<(trainratio+valratio)*len(files):
            fval.write(filename)
            ftrainval.write(filename)
        elif i<(trainratio+valratio+testratio)*len(files):
            ftest.write(filename)
    ftrain.close()
    fval.close()
    ftrainval.close()
    ftest.close()

def generatevocsets(trainratio=0.7,valratio=0.2,testratio=0.1):
    if not os.path.exists(rootdir+"/ImageSets"):
        os.mkdir(rootdir+"/ImageSets")
    if not os.path.exists(rootdir+"/ImageSets/Main"):
        os.mkdir(rootdir+"/ImageSets/Main")
    ftrain=open(rootdir+"/ImageSets/Main/train.txt",'w')
    fval=open(rootdir+"/ImageSets/Main/val.txt",'w')
    ftrainval=open(rootdir+"/ImageSets/Main/trainval.txt",'w')
    ftest=open(rootdir+"/ImageSets/Main/test.txt",'w')
    files=os.listdir(labelsdir)
    for i in range(len(files)):
        imgfilename=files[i][:-4]
        ftrainval.write(imgfilename+"\n")
        if i<int(len(files)*trainratio):
            ftrain.write(imgfilename+"\n")
        elif i<int(len(files)*(trainratio+valratio)):
            fval.write(imgfilename+"\n")
        else:
            ftest.write(imgfilename+"\n")
    ftrain.close()
    fval.close()
    ftrainval.close()
    ftest.close()

if __name__=="__main__":
    showgt()
    generatevocsets()
    generatetxt()
    #generate_hdf5()
    #viewginhdf5()