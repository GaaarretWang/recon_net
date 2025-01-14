from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.functional import interpolate
from torch.nn.functional import pad
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

albedo = getEXR('./REDS/train/albedo/0.exr')#h w c
print('albedo:' , albedo.shape)

depth = getEXR('./REDS/train/depth/0.exr')#h w c
print('depth:' , depth.shape)

normal = getEXR('./REDS/train/normal/0.exr')#h w c
print('normal:' , normal.shape)