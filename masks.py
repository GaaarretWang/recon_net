from typing import Callable, Union, Tuple, List
import torch
import numpy as np
import math

r1 = 75
r2 = 150
scale = 8
p01 = 0.98
p1 = 0.98
p2 = 0.3
p23 =  0.15

def AngularCoord2ScreenCoord(AngularCoord, resolution):
    # transform the angular coords ((0 deg, 0 deg) at screen center) to screen coords which are in the range of
    # 0-1. (0, 0) at Bottom-left, (1, 1) at Top-right
    
    # the parameters of our Hmd (HTC Vive).
    # Vertical FOV.
    VerticalFov = math.pi*91.5/180;
    # Size of a half screen.
    ScreenWidth = resolution[0]
    ScreenHeight = resolution[1]
    # the pixel distance between eye and the screen center.
    ScreenDist = 0.5* ScreenHeight/math.tan(VerticalFov/2);
    
    ScreenCoord = np.zeros(2)
    
    # the X coord.
    ScreenCoord[0] = 0.5 * ScreenWidth + (ScreenDist * math.tan(math.pi*AngularCoord[0] / 180)); 
    # the Y coord.
    ScreenCoord[1] = 0.5 * ScreenHeight + (ScreenDist * math.tan(math.pi*AngularCoord[1] / 180));
    # # flip the Y
    ScreenCoord[1] = ScreenHeight - ScreenCoord[1]
    return ScreenCoord

def move_sample(p_mask, inside, sample_time_input, left):
    zero = torch.zeros_like(p_mask)
    one = torch.ones_like(p_mask)
    sample_sum = torch.sum(torch.where(inside, p_mask, zero)) + left
    nocolor_sample = torch.where(torch.logical_and(inside, sample_time_input > 0.9), one, zero)
    nocolor_sample_sum = torch.sum(nocolor_sample)
    if nocolor_sample_sum > sample_sum:
        nocolor_sample *= (sample_sum / nocolor_sample_sum)
        p_mask = torch.where(torch.logical_and(inside, sample_time_input > 0.9), nocolor_sample, p_mask)
        return p_mask, 0
    else:
        p_mask = torch.where(torch.logical_and(inside, sample_time_input > 0.9), nocolor_sample, p_mask)
        return p_mask, sample_sum - nocolor_sample_sum

def get_center_mask(gaze_points: Tuple[float, float], shape: Tuple[int, int]) -> torch.Tensor:      
    device = gaze_points.device  
    batch_size = gaze_points.shape[0]  
    w, h = shape  

    x_indices, y_indices = torch.meshgrid(  
        torch.arange(w, device=device, dtype=torch.float32),   
        torch.arange(h, device=device, dtype=torch.float32),   
        indexing='xy'  
    )  
    
    indexes = torch.stack((x_indices.flatten(), y_indices.flatten()), dim=0)  
    
    gaze_points = gaze_points.to(torch.float32)  
    
    curAngle_tensor = gaze_points.t().to(device)  
    
    distances = torch.norm(indexes[:, None, :] - curAngle_tensor[:, :, None], p=2, dim=0)  
    distances = distances.reshape(batch_size, h, w)  

    A = torch.tensor([[r1, 1], [r2, 1]], dtype=torch.float32, device=device)  
    b = torch.tensor([[p1], [p2]], dtype=torch.float32, device=device)  
    m_b = torch.linalg.pinv(A) @ b  
    m, b = m_b[0, 0], m_b[1, 0]  

    p_mask = torch.zeros_like(distances, dtype=torch.float32)  
    
    mask_lt_r1 = distances < r1  
    mask_gt_r2 = distances > r2  
    mask_between = (distances >= r1) & (distances <= r2)  
    
    p_mask[mask_lt_r1] = p01  
    p_mask[mask_gt_r2] = p23  
    
    m_float = m if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.float32)  
    b_float = b if isinstance(b, torch.Tensor) else torch.tensor(b, dtype=torch.float32)  
    
    p_mask[mask_between] = m_float * distances[mask_between] + b_float  

    # sample_sums = p_mask.sum(dim=(1, 2), keepdim=True)  
    p_mask *= (1 / torch.mean(p_mask))

    return p_mask.unsqueeze(1)  

def get_mask_two(sample_time_input: torch.Tensor, curAngle: Tuple[float, float], futureAngle: Tuple[float, float], shape: Tuple[int, int]) -> torch.Tensor:
    """
    Method returns a binary fovea mask
    :param new_video: (bool) Flag if a new video is present
    :param shape: (Tuple[int, int]) Image shape
    :return: (torch.Tensor) Fovea mask
    """
    # Get all indexes of image
    indexes = np.stack(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])), axis=0).reshape((2, -1))
    # Make center point
    # center = AngularCoord2ScreenCoord(curAngle, shape_reverse) * shape_reverse
    # center_future = AngularCoord2ScreenCoord(futureAngle, shape_reverse) * shape_reverse
    center = np.array([curAngle[0], curAngle[1]])        # print(center)
    center_future0 = np.array([futureAngle[0], futureAngle[1]])
    center_future1 = np.array([futureAngle[2], futureAngle[3]])
    center_future2 = np.array([futureAngle[4], futureAngle[5]])
    center_future3 = np.array([futureAngle[6], futureAngle[7]])
    center_future4 = np.array([futureAngle[8], futureAngle[9]])
    center_future5 = np.array([futureAngle[10], futureAngle[11]])
    center_future6 = np.array([futureAngle[12], futureAngle[13]])
    center_future7 = np.array([futureAngle[14], futureAngle[15]])
    center_future8 = np.array([futureAngle[16], futureAngle[17]])
    center_future9 = np.array([futureAngle[18], futureAngle[19]])
    # Calc euclidean distances
    distances_cur = torch.from_numpy(np.linalg.norm(indexes - center.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape)
    distances_future = []
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future0.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future1.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future2.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future3.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future4.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future5.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future6.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future7.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future8.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    distances_future.append(torch.from_numpy(np.linalg.norm(indexes - center_future9.reshape((2, 1)), ord=2, axis=0)).to('cuda').reshape(sample_time_input.shape))
    # Calc probability mask

    # current_no_color_sample_num = np.sum(np.where(np.logical_and(no_color_mask.ravel() <  0.5, distances < r1), 1.0, 0.0))
    # future_sample_num = np.sum(np.where(np.logical_and(no_color_mask.ravel() <  0.5, distances_future < r1), 1.0, 0.0))
    # if current_sample_num < future_sample_num:
    #     sample_ratio = 0.5
    # else:
    #     sample_ratio = current_sample_num / (current_sample_num + future_sample_num)
    
    # # sum_index = 0.5
    # # ratio_sum = sum_index * sample_ratio + (1 - sum_index) * normal_ratio
    # sample_m = 2
    # normal_m = 0.01
    # current_min = 0.5
    # current_sample_count = 1 - (normal_ratio - 0.5) ** normal_m / 0.5 ** normal_m * (1 - current_min)
    # global last_sample_count
    # current_sample_count = last_sample_count - current_sample_count
    # if current_sample_count < 0:
    #     current_sample_count = 0
    # current_sample_count = 1 - current_sample_count
    # last_sample_count = current_sample_count
    # print(current_sample_count)
    m, b = np.linalg.pinv(np.array([[r1, 1], [r2, 1]])) @ np.array([[p1], [p2]])
    m = torch.from_numpy(m).to('cuda')
    b = torch.from_numpy(b).to('cuda')
    p_mask = (torch.where((distances_cur < r1), p01, 0.0) \
                + torch.where((distances_cur > r2), p23, 0.0) \
                + torch.where(torch.logical_and(distances_cur >= r1, distances_cur <= r2), m * distances_cur  + b, 0.0)).to(torch.float)
    

    # p_mask, left = move_sample(p_mask, distances_cur < r1, sample_time_input, 0)
    # sampled = distances_cur >= r1
    # for i in range(10):
    #     if left < 0.01:
    #         break;
    #     p_mask, left = move_sample(p_mask, torch.logical_and(distances_future[i] < r1, sampled), sample_time_input, left)
    #     sampled = torch.logical_and(distances_future[i] >= r1, sampled)
    
    # if left > 0.01:
    #     zero = torch.zeros_like(p_mask).to('cuda')
    #     one = torch.ones_like(p_mask).to('cuda')
    #     nosample = torch.where(torch.logical_and(distances_cur < r1, p_mask < 0.9), one, zero)
    #     nosample_sum = torch.sum(nosample)
    #     nosample *= (left / nosample_sum)
    #     p_mask = torch.where(torch.logical_and(distances_cur < r1, p_mask < 0.9), nosample, p_mask)

    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future1 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future1 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future2 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future2 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future3 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future3 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future4 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future4 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future5 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future5 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future6 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future6 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future7 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future7 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future8 < r1, sampled), sample_time_input, left)
    # sampled = torch.logical_and(distances_future8 >= r1, sampled)
    # p_mask, left = self.move_sample(p_mask, torch.logical_and(distances_future9 < r1, sampled), sample_time_input, left)

    # cur_center_sample_sum = np.sum(np.where(distances < r1, p_mask, 0.0))
    # cur_center_nocolor_sample = np.where(np.logical_and((distances < r1), sample_time_input_np > 0.98), 1.0, 0.0)
    # cur_center_nocolor_sample_sum = np.sum(cur_center_nocolor_sample)
    # if cur_center_nocolor_sample_sum > cur_center_sample_sum:
    #     cur_center_nocolor_sample *= (cur_center_sample_sum / cur_center_nocolor_sample_sum)
    #     p_mask = np.where(np.logical_and((distances < r1), sample_time_input_np > 0.98), cur_center_nocolor_sample, p_mask)
    #     return p_mask
    # else:
    #     p_mask = np.where(np.logical_and((distances < r1), sample_time_input_np > 0.98), cur_center_nocolor_sample, p_mask)

    # center_left = cur_center_sample_sum - cur_center_nocolor_sample_sum
    # future_center_sample_sum = np.sum(np.where(np.logical_and(distances_future0 < r1, distances >= r1), p_mask, 0.0))
    # future_totol_sample_sum = future_center_sample_sum + center_left
    # future_center_nocolor_sample = np.where(np.logical_and(distances_future0 < r1, distances >= r1, sample_time_input_np > 0.98), 1.0, 0.0)
    # future_center_nocolor_sample_sum = np.sum(future_center_nocolor_sample)
    # if future_center_nocolor_sample_sum > future_totol_sample_sum:
    #     future_center_nocolor_sample *= (future_totol_sample_sum / future_center_nocolor_sample_sum)
    #     p_mask = np.where(np.logical_and(np.logical_and(distances_future0 < r1, distances >= r1), sample_time_input_np > 0.98), future_center_nocolor_sample, p_mask)
    #     return p_mask
    # else:
    #     p_mask = np.where(np.logical_and(np.logical_and(distances_future0 < r1, distances >= r1), sample_time_input_np > 0.98), future_center_nocolor_sample, p_mask)




    return p_mask


# def get_mask_two(self, sample_time_input_np: np.ndarray, cur_gaze_abs_sub: np.ndarray, future_gaze_abs_sub: np.ndarray, curAngle: Tuple[float, float], futureAngle: Tuple[float, float], shape: Tuple[int, int]) -> torch.Tensor:
#     """
#     Method returns a binary fovea mask
#     :param new_video: (bool) Flag if a new video is present
#     :param shape: (Tuple[int, int]) Image shape
#     :return: (torch.Tensor) Fovea mask
#     """
#     current_index = 1
#     # current_ssim_sub = self.ssim_sub[self.current_index]
#     # if(current_ssim_sub > 0.02):
#     #     current_index = 1 / 2

#     cur_p_mask = self.get_mask(sample_time_input_np, cur_gaze_abs_sub, curAngle, shape)      
#     future_p_mask = self.get_mask(sample_time_input_np, future_gaze_abs_sub, futureAngle, shape)      
#     p_mask = cur_p_mask * current_index + future_p_mask * (1 - current_index)

#     # Make mask
#     return p_mask


def generate_mask(gaze_labels, sample_time) -> torch.Tensor:
    sample_time_input = torch.reciprocal(sample_time + 1)
    sample_time_input = torch.where(sample_time_input > 0.02, sample_time_input, torch.zeros_like(sample_time_input))
    sample_time_input = 1 - sample_time_input

    # save_image(
    #     sample_time_input,
    #     fp=os.path.join(self.path_save_plots,
    #                     'prediction_time_{}_{}.exr'.format(0, self.current_index)),
    #     format='exr',
    #     nrow=self.validation_dataloader.dataset.number_of_frames)

    # pad_size = 80
    # crop_size = 160
    # abs_sub_high = F.interpolate(abs_sub, scale_factor=4, mode='bilinear', align_corners=False)
    # pad_abs_sub = F.pad(abs_sub_high, (pad_size, pad_size, pad_size, pad_size))

    # save_image(
    #     pad_normal,
    #     fp=os.path.join('./tmp',
    #                     'normal_{}.exr'.format(str(datetime.now()).replace(' ', '_').replace(':', '.'))),
    #     format='exr',
    #     nrow=1)
    # black = torch.zeros(abs_sub_high.shape).to('cuda')
    # cur_gaze_abs_sub = black.clone()
    # cur_gaze_abs_sub[:, :, 540-pad_size:540+pad_size, 960-pad_size:960+pad_size] = pad_abs_sub[:, :, 540:540+crop_size, 960:960+crop_size]
    # future_gaze_coord = [max(0, min(int(future_gaze[0]), 1919)), max(0, min(int(future_gaze[1]), 1079))]
    # future_gaze_abs_sub = black.clone()
    # # print(future_gaze_coord)
    # future_gaze_abs_sub[:, :, max(0,future_gaze_coord[1]-pad_size):min(1080, future_gaze_coord[1]+pad_size), max(0,future_gaze_coord[0]-pad_size):min(1920, future_gaze_coord[0]+pad_size)] = pad_abs_sub[:, :, max(0,future_gaze_coord[1]-pad_size)+pad_size:min(1080, future_gaze_coord[1]+pad_size)+pad_size, max(0,future_gaze_coord[0]-pad_size)+pad_size:min(1920, future_gaze_coord[0]+pad_size)+pad_size]
    # future_gaze_abs_sub = pad_abs_sub[:, :, future_gaze_coord[1]:future_gaze_coord[1]+crop_size, future_gaze_coord[0]:future_gaze_coord[0]+crop_size].cpu().numpy()
    # normal_ratio = calculate_variances(cur_gaze_normal, future_gaze_normal)
    # normal_ratio = 0.5
    # time_and_date = str(datetime.now()).replace(' ', '_').replace(':', '.')

    # pred_gaze0_label = gaze_labels[4, :]
    # pred_gaze1_label = gaze_labels[5, :]
    # pred_gaze2_label = gaze_labels[6, :]
    p_mask = get_mask_two(sample_time_input, curAngle=[960, 540], futureAngle=gaze_labels[2:22], 
                            shape=(sample_time.shape[2], sample_time.shape[3]))
    # mask = self.get_mask_two(curAngle=[960, 540], futureAngle=pred_gaze0_label, 
    #                         shape=(label_img.shape[2], label_img.shape[3])).unsqueeze(0).unsqueeze(0).to('cuda')
    # p_mask = self.get_mask(sample_time_input_np, sample_time_input_np, curAngle=[960, 540], shape=(sample_time.shape[2], sample_time.shape[3]))
    # mask = self.get_mask_two(curAngle=[960, 540], futureAngle=torch.tensor(gaze_outputs[batch_num]), 
    #                          shape=(label_img.shape[2], label_img.shape[3])).unsqueeze(0).unsqueeze(0).to('cuda')
    mask = (p_mask >= torch.from_numpy(np.random.uniform(low=0, high=1, size=sample_time.shape[2] * sample_time.shape[3])).reshape(sample_time_input.shape).to('cuda')).to(torch.float)
    # print(mask.shape)
    # save_image(
    #     mask,
    #     fp=os.path.join('./tmp/',
    #                     'prediction_{}.exr'.format(time_and_date)),
    #     format='exr',
    #     nrow=1)
    # save_image(
    #     no_color_mask.unsqueeze(0),
    #     fp=os.path.join('./tmp/',
    #                     'prediction_tmp_{}.exr'.format(time_and_date)),
    #     format='exr',
    #     nrow=1)
    # save_image(
    #     mask,
    #     fp=os.path.join(self.path_save_plots,
    #                     'mask_0_{}.exr'.format(self.current_index)),
    #     format='exr',
    #     nrow=self.validation_dataloader.dataset.number_of_frames)

    return mask

