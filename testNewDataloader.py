
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch
import numpy as np
from typing import Tuple
from torch.utils.data.dataset import Dataset
from torch.nn.functional import interpolate
import torch.nn.functional as F
from torch.nn.functional import pad
def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_mask( shape: Tuple[int, int]) -> torch.Tensor:
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
        [shape[1] - 80,  shape[0] - 50])
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
# Save arguments
number_of_frames = 1
overlapping_frames = 0
# Init previously loaded frames
path = './Bistro/train/train_sharp'
data_path = []
frame_format = 'exr'
# Get all objects in path an search for video folders
for video in sorted(os.listdir(path=path)):

    # Case that object in path is a folder
    if os.path.isdir(os.path.join(path, video)):#/train/000
        # Init frame counter
        frame_counter = 0
        # Init frame index
        frame_index = 0
        # Iterate over all frames in video folder
        while frame_index < len(os.listdir(path=os.path.join(path, video))):#index<len(/train/000)
            # Get current frame name
            current_frame = sorted(os.listdir(path=os.path.join(path, video)))[frame_index]
            # print(current_frame)
            # Check object is a frame of the desired format
            if frame_format in current_frame:
                # Add new list to data path in case of a new frame sequence
                if frame_counter == 0:
                    data_path.append([])
                # Add frame to last data path under list
                data_path[-1].append(os.path.join(path, video, current_frame))
                # Increment frame counter
                frame_counter += 1
                # Reset frame counter if number of frames in one element are reached
                if frame_counter == number_of_frames * 5:#!!!!!!
                    frame_counter = 0
                    # Decrement frame index by the number of overlapping frames
                    frame_index -= overlapping_frames
                # Increment frame index
                frame_index += 1
        # Remove last list element of data_path if number of frames is not matched
        if len(data_path[-1]) != number_of_frames * 5:#!!!!!!
            del data_path[-1]

print(data_path[0])
print('len:',len(data_path))
frames_masked = []
frames_label = []




def get_grid():
    w, h = (256, 192)
    x = torch.linspace(0, w - 1, steps=w)
    y = torch.linspace(0, h - 1, steps=h)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid_x = grid_x
    grid_y = grid_y
    grid_x = grid_x.unsqueeze(2)
    grid_y = grid_y.unsqueeze(2)

    xr = torch.ones(h, w, 1) * (2. / (w - 1))
    yr = torch.ones(h, w, 1) * (2. / (h - 1))
    xyr = torch.cat((xr, yr), dim=2)

    return grid_x, grid_y, w, h, xyr


# def warp(preResult: torch.Tensor, motion: torch.Tensor, curInput: torch.Tensor,
#          device: torch.device = torch.device("cuda:0")):
#     #curInput:c h w
#     #motion:c h w
#     #preResult: c h w
#     print('preResult:',preResult.shape)#3 192 256
#     print('motion:', motion.shape)#2 192 256
#     print('curInput:', curInput.shape)#3 192 256
#     grid_x, grid_y, w, h, xyr = get_grid() #[192, 256, 1]
#     # curInput = curInput.unsqueeze(0)  # 1 c h w
#
#     motion = motion.permute(1,2,0)  # h w c 192 256 2
#     print('new motion:', motion.shape)
#
#     mx, my = torch.split(motion, 1, dim=2)#h w 1:192  256 1
#     print('mx:', mx.shape)
#     print('my:', my.shape)
#
#     grid_mx = grid_x + mx
#     grid_my = grid_y + my
#     print('grid_mx:', grid_mx.shape)#192 256 1
#     print('grid_my:', grid_my.shape)
#     grid = torch.cat((grid_mx, grid_my), dim=2)#192 256 2
#     print('grid:', grid.shape)
#
#     grid = (grid * xyr - 1)#h w 2
#
#     preResult = preResult.unsqueeze(0)#1 c h w
#     grid = grid.unsqueeze(0)
#     warped = F.grid_sample(preResult, grid, align_corners=True)
#     warped = warped.squeeze(0)
#     print('warped_0:',warped.shape)#3 192 256
#
#     outOfBorderX: torch.Tensor = (grid_mx < 0) | (grid_mx >= w)
#     outOfBorderY: torch.Tensor = (grid_my < 0) | (grid_my >= h)
#
#     outOfBorder = (outOfBorderX | outOfBorderY).permute(2,0,1)# hwc->c h w
#
#     print('--------------')
#     print('outOfBorder:',outOfBorder.shape)#1 192 256
#     print('curInput:', curInput.shape)#3 192 256
#     print('warped:', warped.shape)#3 192 256
#     print('--------------')
#     warped = torch.where(outOfBorder, curInput, warped)
#     print('warped_1:', warped.shape)#3 192 256
#     ndarry = warped.permute(1, 2, 0).numpy()
#     cv2.imwrite('./warped.exr', ndarry[:, :, ::-1])  # h w c,所以cv处理必须hwc
#     return warped
def warp(preResult: torch.Tensor, motion: torch.Tensor, curInput: torch.Tensor,
         device: torch.device = torch.device("cuda:0")):
    preResult = preResult.unsqueeze(0)#b c h w
    motion = motion.unsqueeze(0)  # b c h w
    curInput = curInput.unsqueeze(0)  # b c h w

    b, c, h, w = preResult.shape
    x = torch.linspace(-1, 1, steps=w)
    y = torch.linspace(-1, 1, steps=h)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack([grid_x, grid_y], dim=0)

    mx, my = torch.split(motion, 1, dim=1)#channel

    mx_ = mx * 2
    my_ = my * 2

    mv_ = torch.cat([mx_, my_], dim=1)

    gridMV = (grid - mv_).permute(0, 2, 3, 1)
    warped = F.grid_sample(preResult, gridMV, align_corners=True)
    oox, ooy = torch.split((gridMV < -1) | (gridMV > 1), 1, dim=3)
    oo = (oox | ooy).permute(0, 3, 1, 2)

    output = torch.where(oo, curInput, warped)
    output = output.squeeze(0)

    ndarry = output.permute(1, 2, 0).numpy()
    cv2.imwrite('./warped.exr', ndarry[:, :, ::-1])  # h w c,所以cv处理必须hwc
    return output


preResult = getEXR(data_path[20][0])#h w c# 0021_img
preResult = torch.from_numpy(preResult).permute(2,0,1)#c h w 1920*1080->320*180
preResult = interpolate(preResult[None], scale_factor=(1 / 4), mode='bilinear', align_corners=False,
                                recompute_scale_factor=True)[0]
save_pre=preResult.permute(1,2,0).numpy()
cv2.imwrite('./preResult.exr',save_pre[:,:,::-1])


curResult = getEXR(data_path[21][0])#h w c# 0021_img
curResult = torch.from_numpy(curResult).permute(2,0,1)#c h w 1920*1080->320*180
curResult = interpolate(curResult[None], scale_factor=(1 / 4), mode='bilinear', align_corners=False,
                                recompute_scale_factor=True)[0]
save_pre2=curResult.permute(1,2,0).numpy()
cv2.imwrite('./curResult.exr',save_pre2[:,:,::-1])
index = 0
item=21

for frame in data_path[item]:#0022
    print('--------------------------------------------')

    exr_image = getEXR(frame)#h w c
    image = torch.from_numpy(exr_image).permute(2,0,1)#c h w 1920*1080->320*180

    image_low_res = interpolate(image[None], scale_factor=(1 / 4), mode='bilinear', align_corners=False,
                                recompute_scale_factor=True)[0]
    if index %5 ==4:#mv
        if item==0:
            warpedPrev = frames_masked[0]
        else:
            warpedPrev = warp(preResult , image_low_res[:2,:,:] , frames_masked[0])#pre:c h w; mv:2,h,w; cur(label):c,h,w
    else:
        image_low_res_masked = image_low_res * get_mask(shape=(image_low_res.shape[1], image_low_res.shape[2]))#random

        frames_masked.append(image_low_res_masked)
    if index % 5 == 0:

        label_image = interpolate(image[None], scale_factor=(1 / 4), mode='bilinear', align_corners=False,
                                  recompute_scale_factor=True)[0]
        frames_label.append(label_image)#768---------------------
    index = index + 1
    print('frames_masked:',len(frames_masked))
    print('frames_label:', len(frames_label))
# Concatenate frames to tensor of shape (3 * number of frames, height (/ 4), width (/ 4))#c h w c 256 192
frames_masked = torch.cat(frames_masked, dim=0)#frames_masked:0.exr 1.exr 2.exr
frames_label = torch.cat(frames_label, dim=0)#[]
print(frames_masked.shape)
print(frames_label.shape)

