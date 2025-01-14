import os
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
# import cv2
import torch.nn.functional as F
import torch

# frame = './Bistro/train/train_sharp/000/0001_3_depth.exr'
# exr_image = getEXR(frame)#h w c
# image = torch.from_numpy(exr_image).permute(2,0,1)#c h w 1920*1080->320*180
#
# # image = image[:2,:,:]

import torch



albedo = torch.ones((1,3,192,256))
normal = torch.randn((1,3,192,256))
depth = torch.randn((1,3,192,256))
img = torch.randn((1,3,192,256))
input = torch.cat((albedo,normal,depth,img),dim=1)

# mi = img[:,0,:,:].unsqueeze(0)
m = input[:,:3,:,:]
print(input.shape)
print(m)