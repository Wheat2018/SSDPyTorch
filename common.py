"""
    By.Wheat
    2020.05.17
"""
# common import
import os
import cv2
import math
import time
import torch
import numpy as np
import os.path as path
from ssd_face import *
from ssd_face_softmax import *

# common variable
WORK_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(WORK_ROOT, 'data')
WEIGHT_ROOT = os.path.join(WORK_ROOT, 'weights')

use_sigmoid = True

if use_sigmoid:
    SSDType = SSDFace
    SSDLossType = SSDFaceLoss
else:
    SSDType = SSD
    SSDLossType = SSDLoss


