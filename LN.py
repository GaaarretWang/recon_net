import torch
import torch.nn as nn
import torch
import os
from utils import save_image
from datetime import datetime
import torch.nn as nn
from model import DMVModel
from model import TMVModel
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch.nn.functional as F
def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# 创建输入张量，形状为 (1, 3, 192, 256)
input_tensor = torch.randn(1, 3, 192, 256)

# 进行 Layer Normalization
output = F.layer_norm(input_tensor, input_tensor.size()[1:])

# 打印归一化后的张量
print(input_tensor.min(), input_tensor.max())
# 打印输出张量的形状
print(output.min(), output.max())

clamped_tensor = torch.clamp(input_tensor, 0, 1)
print(clamped_tensor.min(), clamped_tensor.max())

one = torch.ones(1,3,192,256)
sin_tensor = torch.sin(input_tensor)
sin_tensor = (sin_tensor + one)/2

print(sin_tensor.min(), sin_tensor.max())
save_image(
                        input_tensor,
                        fp=os.path.join('./LN/input_tensor.exr'),
                        format='exr',
                        nrow=1)

save_image(
                        output,
                        fp=os.path.join('./LN/output.exr'),
                        format='exr',
                        nrow=1)

save_image(
                        clamped_tensor,
                        fp=os.path.join('./LN/clamped_tensor.exr'),
                        format='exr',
                        nrow=1)

save_image(
                        sin_tensor,
                        fp=os.path.join('./LN/sin_tensor.exr'),
                        format='exr',
                        nrow=1)