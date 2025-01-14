import torch
import os
from utils import save_image
from datetime import datetime
import torch.nn as nn
from model import DMVModel
from model import TMVModel
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from torch.nn.functional import interpolate
from torch.nn.functional import pad
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
from typing import Tuple
import cv2
import torch.nn.functional as F
def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_mask(shape: Tuple[int, int]) -> torch.Tensor:
    """
    Method returns a binary fovea mask
    :param new_video: (bool) Flag if a new video is present
    :param shape: (Tuple[int, int]) Image shape
    :return: (torch.Tensor) Fovea mask
    """

    # Get all indexes of image
    indexes = np.stack(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])), axis=0).reshape((2, -1))
    # Make center point
    center = np.array(
        [shape[1] - 80, shape[0] - 50])
    # Calc euclidean distances
    distances = np.linalg.norm(indexes - center.reshape((2, 1)), ord=2, axis=0)
    # Calc probability mask
    m, b = np.linalg.pinv(np.array([[20, 1], [40, 1]])) @ np.array([[0.98], [0.15]])
    p_mask = np.where(distances < 20, 0.98, 0.0) + np.where(distances > 40, 0.15, 0.0) \
                  + np.where(np.logical_and(distances >= 20, distances <= 40), m * distances + b, 0.0)
    # Make mask
    mask = torch.from_numpy(p_mask >= np.random.uniform(low=0, high=1, size=shape[0] * shape[1])).reshape(
        (shape[0], shape[1]))
    return mask.float()

