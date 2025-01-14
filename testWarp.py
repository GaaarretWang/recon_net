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
# w, h = (256, 192)
# x = torch.linspace(0, w - 1, steps=w)
# y = torch.linspace(0, h - 1, steps=h)
# grid_y, grid_x = torch.meshgrid(y, x)
# print('grid_x:' , grid_x.shape)#192,256
# print('grid_y:' , grid_y.shape)#192,256
# grid_x = grid_x.to(device='cuda')
# grid_y = grid_y.to(device='cuda')
# grid_x = grid_x.unsqueeze(2)#([1, 192, 256, 1])    [192, 256, 1]
# grid_y = grid_y.unsqueeze(2)#([1, 192, 256, 1])    [192, 256, 1]
# print('grid_x_new:' , grid_x.shape)
# print('grid_y_new:' , grid_y.shape)
#
# xr = torch.ones(h, w, 1) * (2. / (w - 1))#([1, 192, 256, 1])    [192, 256, 1]
# yr = torch.ones(h, w, 1) * (2. / (h - 1))#([1, 192, 256, 1])    [192, 256, 1]
# xyr = torch.cat((xr, yr), dim=2).to(device='cuda')#([1, 192, 256, 2])   [192, 256, 2]
#
#
#
# print('xr:' , xr.shape)
# print('yr:' , yr.shape)
# print('xyr:' , xyr.shape)
# one = torch.ones((1,1,768,1024))
# warp10 = getEXR('./mvmask/warp/warp_31.exr')
# warp10 = torch.from_numpy(warp10).permute(2,0,1).unsqueeze(0)
# # print(warp10.shape)
# tra10 = getEXR('./mvmask/traMV/5814.exr')
# tra10 = torch.from_numpy(tra10).permute(2,0,1).unsqueeze(0)
# tra10_new = tra10[:,:2,:,:] * 100
# tra10_new = torch.cat((tra10_new,one),dim=1)
# save_image(
#                         tra10_new,
#                         fp=os.path.join('./mvmask/traMV/new_10.exr'),
#                         format='exr',
#                         nrow=1)
# D10 = getEXR('./mvmask/DMV/5814.exr')
# D10 = torch.from_numpy(D10).permute(2,0,1).unsqueeze(0)
# import numpy as np
# mvmask = D10 - tra10
# mvmask = torch.abs(mvmask * 10000)
# mvmask_0 = mvmask[0,0,:,:]
# mvmask_1 = mvmask[0,1,:,:]
# print(mvmask_1.shape)
#------------------------------------------
# mask_0 = torch.where(mvmask_0 > 0.1 , torch.tensor(1.0) , torch.tensor(0.0))
# mask_1 = torch.where(mvmask_1 > 0.1 , torch.tensor(1.0) , torch.tensor(0.0))
# mask =1 -  (mask_0 + mask_1)
#
# print(mask.shape)
# import torchvision
# toPIL = torchvision.transforms.ToPILImage()
# pic = toPIL(mask)
#
# pic.save(f'./mvmask/mask/mask_5814-10000-fan.png')
# # new_warp = warp10*mask
#
# #
# # mask = torch.where((mvmask[:,0,:,:].unsqueeze(0) > 0.1) and (mvmask[:,1,:,:].unsqueeze(0)>0.1), torch.tensor(1.0), torch.tensor(0.0))
# # print(mask.shape)
# #
# # save_image(
# #                         new_warp,
# #                         fp=os.path.join('./mvmask/mask/newWarp.exr'),
# #                         format='exr',
#                         nrow=1)
import numpy as np
from typing import Tuple
from torch.nn.functional import interpolate
import torchvision.transforms as transforms
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
def getMvMask(Dmv :torch.Tensor , tramv :torch.Tensor ):
    Dmv_x = (Dmv[:, 0, :, :] * 1024).unsqueeze(0)
    Dmv_y = (Dmv[:, 1, :, :] * 768).unsqueeze(0)
    Dmv_z = Dmv[:, 2, :, :].unsqueeze(0)
    new_Dmv = torch.cat((Dmv_x, Dmv_y, Dmv_z), dim=1)

    tramv_x = (tramv[:, 0, :, :] * 1024).unsqueeze(0)
    tramv_y = (tramv[:, 1, :, :] * 768).unsqueeze(0)
    tramv_z = tramv[:, 2, :, :].unsqueeze(0)
    new_tramv = torch.cat((tramv_x, tramv_y, tramv_z), dim=1)

    sub = new_Dmv - new_tramv
    sub_x = sub[:, 0, :, :].unsqueeze(0)
    mask_x = torch.where(abs(sub_x) > 0.1, 1., 0.)
    sub_y = sub[:, 1, :, :].unsqueeze(0)
    mask_y = torch.where(abs(sub_y) > 0.1, 1., 0.)

    mask = torch.where(torch.logical_and(mask_x > 0., mask_y > 0.), 1., 0.)

    dilated_tensor = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)


    # device = DMV.device
    # sub = abs((DMV - traMV)*10000).to(device) #1 3 768 1024
    # sub = torch.where(sub > 0.1, 1., 0.).to(device)
    # sub_x = sub[:, 0, :, :].squeeze(0).to(device)
    # sub_y = sub[:, 1, :, :].squeeze(0).to(device)
    # sub = sub_x + sub_y
    # mask = torch.where(abs(sub) > 0., 1., 0.).cpu().numpy()
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    # d_mask = cv2.dilate(mask, kernel, iterations=1)
    #
    # d_mask = torch.from_numpy(1-d_mask).to(device)
    return mask,sub , dilated_tensor
'''
补mv 效果不好，不补了
mask = get_mask((192,256))
mask = mask.reshape((1,1,192,256))
dmvnet = torch.load("./mv/D_model_18.pt")
tmvnet = torch.load("./mv/T_model_18.pt")

Dmv = getEXR('./mv/Dmv.exr')
Dmv = torch.from_numpy(Dmv).permute(2,0,1).unsqueeze(0)#1 3 768 1024
DmvLR = F.interpolate(Dmv, scale_factor=(1 / 4), mode='bilinear',
                                         align_corners=False)#1 3 192 256
maskDmv = DmvLR * mask


tramv = getEXR('./mv/tramv.exr')#1 3 768 1024
tramv = torch.from_numpy(tramv).permute(2,0,1).unsqueeze(0)
TmvLR = F.interpolate(tramv, scale_factor=(1 / 4), mode='bilinear',
                                         align_corners=False)#1 3 192 256
maskTmv = TmvLR * mask

dmv1 = dmvnet(maskDmv[:,0,:,:].unsqueeze(0),mask)
dmv2 = dmvnet(maskDmv[:,1,:,:].unsqueeze(0),mask)
one = torch.ones((1,1,192,256)).to('cuda')
dmv_recon = torch.cat((dmv1,dmv2,one),dim=1)

tmv1 = tmvnet(maskTmv[:,0,:,:].unsqueeze(0),mask)
tmv2 = tmvnet(maskTmv[:,1,:,:].unsqueeze(0),mask)
one = torch.ones((1,1,192,256)).to('cuda')
tmv_recon = torch.cat((tmv1,tmv2,one),dim=1)
'''

Dmv = getEXR('./mv/Dmv.exr')
Dmv = torch.from_numpy(Dmv).permute(2,0,1).unsqueeze(0)#1 3 768 1024
# DmvLR = F.interpolate(Dmv, scale_factor=(1 / 4), mode='bilinear',
#                                          align_corners=False)#1 3 192 256

# DmvLR_HR = F.interpolate(DmvLR, scale_factor=(4), mode='nearest')#1 3 192 256
# save_image(
#                         DmvLR_HR*1000,
#                         fp=os.path.join('./mv/DmvLR_HR.exr'),
#                         format='exr',
#                         nrow=1)
# save_image(
#                         DmvLR*1000,
#                         fp=os.path.join('./mv/DmvLR.exr'),
#                         format='exr',
#                         nrow=1)
# save_image(
#                         Dmv*1000,
#                         fp=os.path.join('./mv/newDmv.exr'),
#                         format='exr',
#                         nrow=1)

tramv = getEXR('./mv/tramv.exr')#1 3 768 1024
tramv = torch.from_numpy(tramv).permute(2,0,1).unsqueeze(0)
# TmvLR = F.interpolate(tramv, scale_factor=(1 / 4), mode='bilinear',
#                                          align_corners=False)#1 3 192 256
# TmvLR_HR = F.interpolate(TmvLR, scale_factor=(4), mode='nearest')#1 3 192 256


mv_mask,sub,dilate = getMvMask(Dmv = Dmv , tramv = tramv)
print(mv_mask.shape)
print('dia:',dilate.shape)



#
warp = getEXR('./mv/5520.exr')
warp = torch.from_numpy(warp).permute(2,0,1).unsqueeze(0)#1 3 768 1024
#
#
dilateWarp = (1-dilate) * warp
#
#
save_image(
                        mv_mask,
                        fp=os.path.join('./mv/mv_mask.exr'),
                        format='exr',
                        nrow=1)
save_image(
                        sub,
                        fp=os.path.join('./mv/sub.exr'),
                        format='exr',
                        nrow=1)
save_image(
                        dilate,
                        fp=os.path.join('./mv/dilate.exr'),
                        format='exr',
                        nrow=1)
save_image(
                        dilateWarp,
                        fp=os.path.join('./mv/dilateWarp.exr'),
                        format='exr',
                        nrow=1)
# save_image(
#                         mv_mask,
#                         fp=os.path.join('./mv/mv_mask.exr'),
#                         format='exr',
#                         nrow=1)
