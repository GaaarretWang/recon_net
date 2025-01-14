import torch
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch.nn.functional as F
def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb
from utils import save_image
def sRGBGamma(tensor):
    threshold = 0.0031308
    a = 0.055
    mult = 12.92
    gamma = 2.4
    res = torch.zeros_like(tensor)
    mask = tensor > threshold
    # image_lo = tensor * mult
    # 0.001 is to avoid funny thing at 0.
    # image_hi = (1 + a) * torch.pow(tensor + 0.001, 1.0 / gamma) - a
    res[mask] = (1 + a) * torch.pow(tensor[mask] + 0.001, 1.0 / gamma) - a

    res[~mask] = tensor[~mask] * mult
    # return mask * image_hi + (1 - mask) * image_lo
    return res

img = getEXR('./5500.exr')
img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
gamma = sRGBGamma(img)
save_image(
                        gamma,
                        fp=os.path.join('./5500-gamma.png'),
                        format='png',
                        nrow=1)