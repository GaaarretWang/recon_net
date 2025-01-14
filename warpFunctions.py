from typing import Callable, Union, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np
import pytorch_ssim
import cv2
import math
from numba import cuda
from masks import generate_mask
from utils import save_image
from datetime import datetime
import os

def warp(scale, grid: torch.Tensor, grid_norm: torch.Tensor, pre_color: torch.Tensor, pre_albedo: torch.Tensor, high_albedo: torch.Tensor, pre_normal: torch.Tensor, high_normal: torch.Tensor, mv: torch.Tensor, cur: torch.Tensor, sample_time: torch.Tensor, gaze_labels: torch.Tensor, high_objectId: torch.Tensor, warped_id: torch.Tensor):
    b, c, h, w = pre_color.shape # 1 3 h w

    gridMV = (grid + 2 * mv)
    grid = gridMV.clone()

    time_and_date = str(datetime.now()).replace(' ', '_').replace(':', '.')
    one_tensor = torch.ones([grid.shape[0], 1, grid.shape[2], grid.shape[3]]).to('cuda')
    # save_image(
    #     torch.cat((mv, one_tensor), dim=1),
    #     fp=os.path.join("ssim_tmp",
    #                     'grid_before_{}.exr'.format(time_and_date)),
    #     format='exr',
    #     nrow=1)

    warped_color, grid, sample_time, warped_id, warped_albedo, warped_normal = grid_sample(pre_color, grid, grid_norm, gridMV, sample_time, warped_id, high_objectId, pre_albedo, pre_normal)
    
    # save_image(
    #     1 / (sample_time + 1.0),
    #     fp=os.path.join(self.path_save_plots,
    #                     'prediction_warped_sample_time_{}.exr'.format(self.current_index)),
    #     format='exr',
    #     nrow=self.validation_dataloader.dataset.number_of_frames)
    # save_image(
    #     torch.cat((warped_id, one_tensor, one_tensor), dim=1),
    #     fp=os.path.join("ssim_tmp",
    #                     'after_warped_id_{}.exr'.format(time_and_date)),
    #     format='exr',
    #     nrow=1)

    

    high_mask = generate_mask(gaze_labels[0], sample_time)
    # high_mask = moveMask(high_mask, sample_time)
    # save_image(
    #     high_mask,
    #     fp=os.path.join("ssim_tmp",
    #                     'mask_{}.exr'.format(time_and_date)),
    #     format='exr',
    #     nrow=1)
    # save_image(
    #     warped_color,
    #     fp=os.path.join("ssim_tmp",
    #                     'sampled_color_{}.exr'.format(time_and_date)),
    #     format='exr',
    #     nrow=1)
    # save_image(
    #     cur,
    #     fp=os.path.join(self.path_save_plots,
    #                     'prediction_cur_mask_{}.exr'.format(self.current_index)),
    #     format='exr',
    #     nrow=self.validation_dataloader.dataset.number_of_frames)

    sample_time = (sample_time + 1) * torch.abs(1 - high_mask)
    warped_id = torch.where(high_mask > 0.5, high_objectId, warped_id)

    # oox, ooy = torch.split((gridMV < -1) | (gridMV > 1), 1, dim=1)
    # oo = (oox | ooy)
    # pre_p = warped_color.permute(0, 2, 3, 1) #[1, 768, 1024, 3]
    # pre_sum = pre_p.sum(dim=3, keepdim=True).squeeze(3)
    # pre_zero = (pre_sum < 0.0001).unsqueeze(3).permute(0, 3, 1, 2)
    # cur_p = cur.permute(0, 2, 3, 1) #[1, 768, 1024, 3]
    # cur_sum = cur_p.sum(dim=3, keepdim=True).squeeze(3)
    # cur_zero = (cur_sum < 0.0001).unsqueeze(3).permute(0, 3, 1, 2)
    
    # save_image(
    #     cur + torch.abs(1 - high_mask) * warped_color,
    #     fp=os.path.join(self.path_save_plots,
    #                     'prediction_warp+cur_{}.exr'.format(self.current_index)),
    #     format='exr',
    #     nrow=self.validation_dataloader.dataset.number_of_frames)

    return cur * high_mask + torch.abs(1 - high_mask) * warped_color,\
        grid_norm * high_mask + torch.abs(1 - high_mask) * grid,\
        sample_time, warped_id,\
        high_albedo * high_mask + torch.abs(1 - high_mask) * warped_albedo,\
        high_normal * high_mask + torch.abs(1 - high_mask) * warped_normal
        # torch.where(oo | pre_zero, grid_norm, torch.where(cur_zero, grid, grid_norm)),\

def grid_sample(pre_color, grid, grid_norm, gridMV, sample_time, warped_id, high_objectId, pre_albedo, pre_normal):
    b, c, h, w = pre_color.shape
    deltaW = 2 / w
    deltaH = 2 / h
    pixel_pos = torch.zeros(gridMV.shape).to('cuda')
    pixel_pos[:, 0, :, :] = gridMV[:, 0, :, :] / deltaW + w / 2
    pixel_pos[:, 1, :, :] = gridMV[:, 1, :, :] / deltaH + h / 2

    output_color = torch.zeros(pre_color.shape).to('cuda')
    output_grid = grid_norm.clone().to('cuda')
    output_sample_time = torch.full(sample_time.shape, 500).to('cuda')
    output_warped_id = torch.zeros(warped_id.shape).to('cuda')
    output_albedo = torch.zeros(pre_albedo.shape).to('cuda')
    output_normal = torch.zeros(pre_normal.shape).to('cuda')
    # lock = torch.zeros(sample_time.shape).numpy().astype(np.int32)
    lock = torch.zeros(sample_time.shape, dtype=torch.int32).to('cuda')

    gpu_grid_sample[(int)(grid_norm.shape[2] * grid_norm.shape[3] / 1024 + 1), 1024](pre_color, grid, pixel_pos, output_color, output_grid, sample_time, output_sample_time, warped_id, high_objectId, output_warped_id, pre_albedo, output_albedo, pre_normal, output_normal, lock)

    return output_color, output_grid, output_sample_time, output_warped_id, output_albedo, output_normal


@cuda.jit
def gpu_grid_sample(pre_color, grid, pixel_pos, output_color, output_grid, sample_time, output_sample_time, warped_id, high_objectId, output_warped_id, pre_albedo, output_albedo, pre_normal, output_normal, lock: np.ndarray):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= pixel_pos.shape[2] * pixel_pos.shape[3]:
        return
    column = idx // pixel_pos.shape[3]
    row = idx % pixel_pos.shape[3]
    mv_w_int = int(pixel_pos[0, 0, column, row])
    mv_h_int = int(pixel_pos[0, 1, column, row])

    if (sample_time[0, 0, column, row] > 999):
        return
    if ((mv_w_int >= pixel_pos.shape[3]) | (mv_w_int < 0) | (mv_h_int < 0) | (mv_h_int >= pixel_pos.shape[2])):
        return
    if (sample_time[0, 0, column, row] >= output_sample_time[0, 0, mv_h_int, mv_w_int]):
        return
    # eps = 0.1
    # if ((pre_depth[0, 0, column, row] < cur_depth[0, 0, mv_h_int, mv_w_int] * (1-eps)) | (pre_depth[0, 0, column, row] > cur_depth[0, 0, mv_h_int, mv_w_int] * (1+eps))):
    #     return

    if warped_id[0, 0, column, row] != high_objectId[0, 0, mv_h_int, mv_w_int]:
        return

    while cuda.atomic.compare_and_swap(lock[0, :, mv_h_int, mv_w_int], 0, 1) == 1:
        pass
    output_sample_time[0, 0, mv_h_int, mv_w_int] = sample_time[0, 0, column, row]
    output_warped_id[0, 0, mv_h_int, mv_w_int] = warped_id[0, 0, column, row]
    for i in range(3):
        output_color[0, i, mv_h_int, mv_w_int] = pre_color[0, i, column, row]
        output_albedo[0, i, mv_h_int, mv_w_int] = pre_albedo[0, i, column, row]
        output_normal[0, i, mv_h_int, mv_w_int] = pre_normal[0, i, column, row]
    for i in range(2):
        output_grid[0, i, mv_h_int, mv_w_int] = grid[0, i, column, row]
    lock[0, 0, mv_h_int, mv_w_int] = 0

@cuda.jit
def pre_process(pre_color, cur_albedo, pre_albedo, cur_normal, pre_normal, out_color, out_albedo, out_normal, sample_time_input, sample_time_output, deltas):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= pre_color.shape[2] * pre_color.shape[3]:
        return
    column = idx // pre_color.shape[3]
    row = idx % pre_color.shape[3]


    diff_min = 100.0
    for i in range(9):
        deltaX = deltas[i][0]
        deltaY = deltas[i][1]
        albedo_diff = 0
        normal_diff = 0
        if 0 <= (column + deltaY) < pre_albedo.shape[2] and 0 <= (row + deltaX) < pre_albedo.shape[3]:

            for channel in range(3):  # Iterate through each channel
                if i == 0:
                    albedo_diff += abs(cur_albedo[0, channel, column, row] - pre_albedo[0, channel, column, row])
                    normal_diff += abs(cur_normal[0, channel, column, row] - pre_normal[0, channel, column, row])
                else:
                    albedo_diff += abs(cur_albedo[0, channel, column + deltaY, row + deltaX] - cur_albedo[0, channel, column, row])
                    normal_diff += abs(cur_normal[0, channel, column + deltaY, row + deltaX] - cur_normal[0, channel, column, row])
                    albedo_diff += abs(cur_albedo[0, channel, column + deltaY, row + deltaX] - pre_albedo[0, channel, column + deltaY, row + deltaX])
                    normal_diff += abs(cur_normal[0, channel, column + deltaY, row + deltaX] - pre_normal[0, channel, column + deltaY, row + deltaX])

            
            if diff_min > max(albedo_diff, normal_diff): 
                diff_min = max(albedo_diff, normal_diff)
                for channel in range(3):  # Iterate through each channel
                    out_color[0, channel, column, row] = pre_color[0, channel, column + deltaY, row + deltaX]
                    out_albedo[0, channel, column, row] = pre_albedo[0, channel, column + deltaY, row + deltaX]
                    out_normal[0, channel, column, row] = pre_normal[0, channel, column + deltaY, row + deltaX]

                sample_time_output[0, 0, column, row] = sample_time_input[0, 0, column + deltaY, row + deltaX]
                if diff_min < 0.03 and i == 0:
                    return

















# not used

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
    
def warpFuntion(preResult: torch.Tensor, motion: torch.Tensor, curInput: torch.Tensor,
    device: torch.device = torch.device("cuda:0")):
    # preResult = preResult.unsqueeze(0)  # b c h w

    device = preResult.device
    motion = motion.to(device)  # b c h w
    # print('motion:',motion.shape)
    curInput = curInput.to(device)  # b c h w
    # print('curInput:', curInput.shape)
    b, c, h, w = preResult.shape
    x = torch.linspace(-1, 1, steps=w)
    y = torch.linspace(-1, 1, steps=h)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack([grid_x, grid_y], dim=0).to(device)

    mx, my = torch.split(motion, 1, dim=1)  # channel

    mx_ = mx * 2
    my_ = my * 2

    mv_ = torch.cat([mx_, my_], dim=1).to(device)

    gridMV = (grid - mv_).permute(0, 2, 3, 1)  # .to
    warped = F.grid_sample(preResult, gridMV, align_corners=True)  # .to
    oox, ooy = torch.split((gridMV < -1) | (gridMV > 1), 1, dim=3)
    oo = (oox | ooy).permute(0, 3, 1, 2)  # .to

    output = torch.where(oo, warped, warped)
    # output = output.squeeze(0)

    # ndarry = output.permute(1, 2, 0).numpy()
    # cv2.imwrite('./warped.exr', ndarry[:, :, ::-1])  # h w c,所以cv处理必须hwc
    return output

def getMvMask(Dmv :torch.Tensor , tramv :torch.Tensor,width:int = 1920,height: int = 1080 ):#1 3 192  256
    Dmv_x = (Dmv[:, 0, :, :] * width).unsqueeze(0)
    Dmv_y = (Dmv[:, 1, :, :] * height).unsqueeze(0)
    Dmv_z = Dmv[:, 2, :, :].unsqueeze(0)
    new_Dmv = torch.cat((Dmv_x, Dmv_y, Dmv_z), dim=1)

    tramv_x = (tramv[:, 0, :, :] * width).unsqueeze(0)
    tramv_y = (tramv[:, 1, :, :] * height).unsqueeze(0)
    tramv_z = tramv[:, 2, :, :].unsqueeze(0)
    new_tramv = torch.cat((tramv_x, tramv_y, tramv_z), dim=1)

    sub = new_Dmv - new_tramv
    sub_x = sub[:, 0, :, :].unsqueeze(0)
    mask_x = torch.where(abs(sub_x) > 0.1, 1., 0.)
    sub_y = sub[:, 1, :, :].unsqueeze(0)
    mask_y = torch.where(abs(sub_y) > 0.1, 1., 0.)

    mask_tensor = mask_x + mask_y #1 1 768 1024
    mask_tensor = torch.where(mask_tensor>0. , 1. , 0.)

    mask_numpy = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()#768 1024   192 256

    # mask = torch.where(torch.logical_and(mask_x > 0., mask_y > 0.), 1., 0.)
    # dilated_tensor = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    # device = DMV.device
    # sub = abs((DMV - traMV)*10000).to(device) #1 3 768 1024
    # sub = torch.where(sub > 0.1, 1., 0.).to(device)
    # sub_x = sub[:, 0, :, :].squeeze(0).to(device)
    # sub_y = sub[:, 1, :, :].squeeze(0).to(device)
    # sub = sub_x + sub_y
    # mask = torch.where(abs(sub) > 0., 1., 0.).cpu().numpy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    d_mask = cv2.dilate(mask_numpy, kernel, iterations=1)
    #
    d_mask = torch.from_numpy(d_mask).unsqueeze(0).unsqueeze(0).to(Dmv.device)
    return 1-d_mask

def calculate_variances(cur_normal, future_normal):
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to('cuda')
    w1 = torch.Tensor(np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).reshape(1, 1, 3, 3)).to('cuda')
    conv1.weight = nn.Parameter(w1)

    conv_cur_normal = torch.zeros_like(cur_normal)
    conv_future_normal = torch.zeros_like(future_normal)
    for i in range(3):
        conv_cur_normal[:, i, :, :] = conv1(conv1(cur_normal[:, i:i+1, :, :]))
        conv_future_normal[:, i, :, :] = conv1(conv1(future_normal[:, i:i+1, :, :]))

    cur_normal_ssim = pytorch_ssim.ssim(conv_cur_normal, cur_normal).item()
    future_normal_ssim = pytorch_ssim.ssim(conv_future_normal, future_normal).item()
    # print(cur_normal_ssim)
    # print(future_normal_ssim)

    # if cur_total_variance + future_total_variance < 0.001:
    #     return 0.6
    # save_image(
    #     torch.abs(cur_normal - conv_cur_normal),
    #     fp=os.path.join('./tmp',
    #                     'abs_sub_{}.exr'.format(str(datetime.now()).replace(' ', '_').replace(':', '.'))),
    #     format='exr',
    #     nrow=1)
    # save_image(
    #     cur_normal,
    #     fp=os.path.join('./tmp',
    #                     'cur_{}.exr'.format(str(datetime.now()).replace(' ', '_').replace(':', '.'))),
    #     format='exr',
    #     nrow=1)
    # save_image(
    #     conv_cur_normal,
    #     fp=os.path.join('./tmp',
    #                     'conv_{}.exr'.format(str(datetime.now()).replace(' ', '_').replace(':', '.'))),
    #     format='exr',
    #     nrow=1)
    normal_ratio = cur_normal_ssim / (cur_normal_ssim + future_normal_ssim)
    if normal_ratio < 0.5:
        return 0.5
    else:
        return normal_ratio


    cur_channel_variances = torch.var(cur_normal, dim=(2, 3))
    cur_total_variance = torch.sum(cur_channel_variances).item()
    future_channel_variances = torch.var(future_normal, dim=(2, 3))
    future_total_variance = torch.sum(future_channel_variances).item()
    if cur_total_variance + future_total_variance < 0.001:
        return 0.6
    normal_ratio = future_total_variance / (cur_total_variance + future_total_variance)
    if normal_ratio < 0.6:
        return 0.6
    else:
        return normal_ratio

@cuda.jit
def moveMaskGpu(mask, sample_time, moveDir, output_mask):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= mask.shape[2] * mask.shape[3]:
        return
    column = idx // mask.shape[3]
    row = idx % mask.shape[3]

    if mask[0, 0, column, row] < 0.5:
        return
    output_mask[0, 0, column, row] = 1

    dst_column = column + moveDir[1]
    dst_row = row + moveDir[0]

    if dst_column < 0 | dst_column >= mask.shape[2]:
        return
    if dst_row < 0 | dst_row >= mask.shape[3]:
        return
    if mask[0, 0, dst_column, dst_row] > 0.5:
        return
    if sample_time[0, 0, column, row] > 16:
        return
    if(sample_time[0, 0, column, row] < sample_time[0, 0, dst_column, dst_row]):
        output_mask[0, 0, column, row] = 0
        output_mask[0, 0, dst_column, dst_row] = 1


def moveMask(mask, sample_time):
    moveDirs = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]).to('cuda')
    for i in range(0, 8):
        # dir_index = torch.randint(0, 8, (1,))
        dir_index = i
        output_mask = torch.zeros(mask.shape).to('cuda')
        moveMaskGpu[(int)(mask.shape[2] * mask.shape[3] / 1024 + 1), 1024](mask, sample_time, moveDirs[dir_index].squeeze(0), output_mask)
        mask = output_mask

    return output_mask



@cuda.jit
def gen_mipmap(pre_recon, pre_recon_2):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= pre_recon_2.shape[2] * pre_recon_2.shape[3]:
        return
    column = idx // pre_recon_2.shape[3]
    row = idx % pre_recon_2.shape[3]

    r = 0
    g = 0
    b = 0
    count = 0
    for i in range(2):
        for j in range(2):
            if column * 2 + i < pre_recon.shape[2] and row * 2 + j < pre_recon.shape[3]:
                if pre_recon[0, 0, column * 2 + i, row * 2 + j] + pre_recon[0, 1, column * 2 + i, row * 2 + j] + pre_recon[0, 2, column * 2 + i, row * 2 + j] > 0.0001:
                    r += pre_recon[0, 0, column * 2 + i, row * 2 + j]
                    g += pre_recon[0, 1, column * 2 + i, row * 2 + j]
                    b += pre_recon[0, 2, column * 2 + i, row * 2 + j]
                    count += 1
    if count > 0:
        pre_recon_2[0, 0, column, row] = r / count
        pre_recon_2[0, 1, column, row] = g / count
        pre_recon_2[0, 2, column, row] = b / count

@cuda.jit
def recon_mipmap(pre_recon, pre_recon_2, pre_recon_4, pre_recon_8, pre_recon_16):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= pre_recon.shape[2] * pre_recon.shape[3]:
        return
    column = idx // pre_recon.shape[3]
    row = idx % pre_recon.shape[3]

    if pre_recon[0, 0, column, row] + pre_recon[0, 1, column, row] + pre_recon[0, 2, column, row] > 0.0001:
        return
    if pre_recon_2[0, 0, column // 2, row // 2] + pre_recon_2[0, 1, column // 2, row // 2] + pre_recon_2[0, 2, column // 2, row // 2] > 0.0001:
        pre_recon[0, 0, column, row] = pre_recon_2[0, 0, column // 2, row // 2]
        pre_recon[0, 1, column, row] = pre_recon_2[0, 1, column // 2, row // 2]
        pre_recon[0, 2, column, row] = pre_recon_2[0, 2, column // 2, row // 2]
        return
    if pre_recon_4[0, 0, column // 4, row // 4] + pre_recon_4[0, 1, column // 4, row // 4] + pre_recon_4[0, 2, column // 4, row // 4] > 0.0001:
        pre_recon[0, 0, column, row] = pre_recon_4[0, 0, column // 4, row // 4]
        pre_recon[0, 1, column, row] = pre_recon_4[0, 1, column // 4, row // 4]
        pre_recon[0, 2, column, row] = pre_recon_4[0, 2, column // 4, row // 4]
        return
    if pre_recon_8[0, 0, column // 8, row // 8] + pre_recon_8[0, 1, column // 8, row // 8] + pre_recon_8[0, 2, column // 8, row // 8] > 0.0001:
        pre_recon[0, 0, column, row] = pre_recon_8[0, 0, column // 8, row // 8]
        pre_recon[0, 1, column, row] = pre_recon_8[0, 1, column // 8, row // 8]
        pre_recon[0, 2, column, row] = pre_recon_8[0, 2, column // 8, row // 8]
        return
    if pre_recon_16[0, 0, column // 16, row // 16] + pre_recon_16[0, 1, column // 16, row // 16] + pre_recon_16[0, 2, column // 16, row // 16] > 0.0001:
        pre_recon[0, 0, column, row] = pre_recon_16[0, 0, column // 16, row // 16]
        pre_recon[0, 1, column, row] = pre_recon_16[0, 1, column // 16, row // 16]
        pre_recon[0, 2, column, row] = pre_recon_16[0, 2, column // 16, row // 16]
        return

@cuda.jit
def recon(color, depth, out_color, sample_time, kernel_size, eps_size):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= color.shape[2] * color.shape[3]:
        return
    column = idx // color.shape[3]
    row = idx % color.shape[3]

    if(color[0, 0, column, row] + color[0, 1, column, row] + color[0, 2, column, row] > 0.00001):
        for i in range(3):
            out_color[0, i, column, row] = color[0, i, column, row]
        return

    
    count = 0
    sigma = 1
    eps = eps_size
    for i in range(-kernel_size, kernel_size + 1):
        for j in range(-kernel_size, kernel_size + 1):
            if i == 0 and j == 0:
                continue
            neighbor_column, neighbor_row = column + i, row + j
            if 0 <= neighbor_column < color.shape[2] and 0 <= neighbor_row < color.shape[3]:
                if(color[0, 0, neighbor_column, neighbor_row] + color[0, 1, neighbor_column, neighbor_row] + color[0, 2, neighbor_column, neighbor_row] > 0.001):
                    if((depth[0, 0, neighbor_column, neighbor_row] > depth[0, 0, column, row] * (1-eps)) & (depth[0, 0, neighbor_column, neighbor_row] < depth[0, 0, column, row] * (1+eps))):
                        distance = math.exp(-(j**2 + i**2) / (2 * sigma**2))
                        for i in range(3):
                            out_color[0, i, column, row] += color[0, i, neighbor_column, neighbor_row] * distance / (sample_time[0, 0, neighbor_column, neighbor_row] + 1)
                        count += distance / (sample_time[0, 0, neighbor_column, neighbor_row] + 1)
    if count != 0:
        for i in range(3):
            out_color[0, i, column, row] /= count

def reconstruct(pre_features_warped, depth, sample_time):
    b, c, h, w = pre_features_warped.shape
    recon_result = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](pre_features_warped, depth, recon_result, sample_time, 1, 0.001)
    recon_result1 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result, depth, recon_result1, sample_time, 1, 0.001)
    recon_result2 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result1, depth, recon_result2, sample_time, 1, 0.02)
    recon_result3 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result2, depth, recon_result3, sample_time, 1, 0.02)
    recon_result4 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result3, depth, recon_result4, sample_time, 2, 0.02)
    recon_result5 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result4, depth, recon_result5, sample_time, 3, 0.02)
    recon_result6 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result5, depth, recon_result6, sample_time, 4, 0.02)
    recon_result7 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result6, depth, recon_result7, sample_time, 4, 0.03)
    recon_result8 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result7, depth, recon_result8, sample_time, 4, 0.03)
    recon_result9 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result8, depth, recon_result9, sample_time, 4, 0.03)
    recon_result10 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result9, depth, recon_result10, sample_time, 4, 0.03)
    recon_result11 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result10, depth, recon_result11, sample_time, 4, 0.03)
    recon_result12 = torch.zeros(pre_features_warped.shape).to('cuda')
    recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 32 + 1), 32](recon_result11, depth, recon_result12, sample_time, 4, 0.03)
    # mipmap2 = torch.zeros([b, c, h//2, w//2]).to('cuda')
    # gen_mipmap[(int)(mipmap2.shape[2] * mipmap2.shape[3] / 32 + 1), 32](recon_result7, mipmap2)
    # mipmap4 = torch.zeros([b, c, h//4, w//4]).to('cuda')
    # gen_mipmap[(int)(mipmap4.shape[2] * mipmap4.shape[3] / 32 + 1), 32](mipmap2, mipmap4)
    # mipmap8 = torch.zeros([b, c, h//8, w//8]).to('cuda')
    # gen_mipmap[(int)(mipmap8.shape[2] * mipmap8.shape[3] / 32 + 1), 32](mipmap4, mipmap8)
    # mipmap16 = torch.zeros([b, c, h//16, w//16]).to('cuda')
    # gen_mipmap[(int)(mipmap16.shape[2] * mipmap16.shape[3] / 32 + 1), 32](mipmap8, mipmap16)
    # recon_mipmap[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 1024 + 1), 1024](recon_result7, mipmap2, mipmap4, mipmap8, mipmap16)

    # recon_result3 = torch.zeros(pre_features_warped.shape).to('cuda')
    # recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 1024 + 1), 1024](recon_result2, depth, recon_result3, sample_time, 2)
    # recon_result4 = torch.zeros(pre_features_warped.shape).to('cuda')
    # recon[(int)(pre_features_warped.shape[2] * pre_features_warped.shape[3] / 1024 + 1), 1024](recon_result3, depth, recon_result4, sample_time, 10)
    return recon_result12
