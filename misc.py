from typing import Tuple, Union, List

import os
import json
import torch
import numpy as np

import torchvision.transforms.functional as tf
from torch.nn.functional import interpolate
from PIL import Image
from torch.nn.functional import pad
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

class Logger(object):
    """
    Class to log different metrics
    Source: https://github.com/ChristophReich1996/Semantic_Pyramid_for_Image_Generation
    """

    def __init__(self) -> None:
        self.metrics = dict()
        self.hyperparameter = dict()

    def log(self, metric_name: str, value: float) -> None:
        """
        Method writes a given metric value into a dict including list for every metric
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def save_metrics(self, path: str) -> None:
        """
        Static method to save dict of metrics
        :param metrics: (Dict[str, List[float]]) Dict including metrics
        :param path: (str) Path to save metrics
        :param add_time_to_file_name: (bool) True if time has to be added to filename of every metric
        """
        # Save dict of hyperparameter as json file
        with open(os.path.join(path, 'hyperparameter.txt'), 'w') as json_file:
            json.dump(self.hyperparameter, json_file)
        # Iterate items in metrics dict
        for metric_name, values in self.metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            torch.save(values, os.path.join(path, '{}.pt'.format(metric_name)))


def psnr(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Function computes the Peak Signal to Noise Ratio
    PSNR = 10 * log10(max[y]**2 / MSE(y, y'))
    Source: https://github.com/ChristophReich1996/CellFlowNet
    :param prediction: (torch.Tensor) Prediction
    :param label: (torch.Tensor) Label
    :return: (torch.Tensor) PSNR value
    """
    assert prediction.numel() == label.numel(), 'Prediction tensor and label tensor must have the number of elements'
    return 10.0 * torch.log10(prediction.max() ** 2 / (torch.mean((prediction - label) ** 2) + 1e-08))


def ssim(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Function computes the structural similarity
    Source: https://github.com/ChristophReich1996/CellFlowNet
    :param prediction: (torch.Tensor) Prediction
    :param label: (torch.Tensor) Label
    :return: (torch.Tensor) SSMI value
    """
    assert prediction.numel() == label.numel(), 'Prediction tensor and label tensor must have the number of elements'
    # Calc means and vars
    prediction_mean = prediction.mean()
    prediction_var = prediction.var()
    label_mean = label.mean()
    label_var = label.var()
    # Calc correlation coefficient
    correlation_coefficient = (1 / label.numel()) * torch.sum((prediction - prediction_mean) * (label - label_mean))
    return ((2.0 * prediction_mean * label_mean) * (2.0 * correlation_coefficient)) / \
           ((prediction_mean ** 2 + label_mean ** 2) * (prediction_var + label_var))


def normalize_0_1_batch(input: torch.tensor) -> torch.tensor:
    '''
    Normalize a given tensor to a range of [0, 1]
    Source: https://github.com/ChristophReich1996/Semantic_Pyramid_for_Image_Generation/blob/master/misc.py

    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    '''
    input_flatten = input.view(input.shape[0], -1)
    return ((input - torch.min(input_flatten, dim=1)[0][:, None, None, None]) / (
            torch.max(input_flatten, dim=1)[0][:, None, None, None] -
            torch.min(input_flatten, dim=1)[0][:, None, None, None]))


def get_fovea_mask(shape: Tuple[int, int], p_mask: torch.Tensor = None, return_p_mask=True) -> Union[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Function generators a fovea mask for a given probability mask. If no p. mask is given the p. mask is also produced.
    :param shape: (Tuple[int, int]) Shape of the final mask
    :param p_mask: (torch.Tensor) Probability mask
    :param return_p_mask: (bool) If true the probability mask will also be returned
    :return: (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) Fovea mask and optional p. mask additionally
    """
    if p_mask is None:
        # Get all indexes of image
        indexes = np.stack(np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0])), axis=0).reshape((2, -1))
        # Make center point
        # center = np.array(
        #     [np.random.uniform(50, shape[1] - 50), np.random.uniform(50, shape[0] - 50)])
        center = np.array(
            [shape[1] - 80, shape[0] - 50])
        # Calc euclidean distances
        distances = np.linalg.norm(indexes - center.reshape((2, 1)), ord=2, axis=0)
        # Calc probability mask
        m, b = np.linalg.pinv(np.array([[20, 1], [45, 1]])) @ np.array([[0.98], [0.15]])
        p_mask = np.where(distances < 20, 0.98, 0.0) + np.where(distances > 40, 0.15, 0.0) \
                 + np.where(np.logical_and(distances >= 20, distances <= 40), m * distances + b, 0.0)
        # Probability mask to torch tesnor
        p_mask = torch.from_numpy(p_mask)
    # Make randomized fovea mask
    mask = torch.from_numpy(p_mask.numpy() >= np.random.uniform(low=0, high=1, size=shape[0] * shape[1])).reshape(
        (shape[0], shape[1]))
    if return_p_mask:
        return mask.float(), p_mask.float()
    return mask.float()


def load_inference_data(path: str) -> List[torch.Tensor]:
    """
    Function loads inference data
    :param path: (str) Path to inference data folder
    :return: (List[torch.Tensor]) List of sequences with the shape of [1 (batch size), 3 * 6 (rgb * frames), 192, 256]
    """

    number_of_frames=2
    overlapping_frames=0
    data_path=[] #列表,每个元素是一个序列，每个序列包括六个帧的地址！
    for video in sorted(os.listdir(path=path)):
        if os.path.isdir(os.path.join(path,video)):#如果video也是各文件夹
            frame_count=0
            frame_index=0
            while frame_index < len(os.listdir(path=os.path.join(path,video))):#遍历文件夹里的每一帧
                current_frame = sorted(os.listdir(path=os.path.join(path, video)))[frame_index]  # 获取当前索引对应文件名,即000000.png
                if frame_count==0:
                    data_path.append([])
                data_path[-1].append(os.path.join(path,video,current_frame))#把帧的路径存进去
                frame_count +=1
                if frame_count == number_of_frames * 4:
                    frame_count=0
                    frame_index -= overlapping_frames

                frame_index += 1

            if len(data_path[-1]) != number_of_frames * 4:
                del data_path[-1]

    #print(data_path)

    infer_data= []
    for sequence in data_path:#遍历每个序列
        frames_low_res = []
        for frame in sequence:#遍历每个帧
            exr_image = getEXR(frame)  # h w c
            image = torch.from_numpy(exr_image).permute(2, 0, 1)  # c h w 1920*1080->320*180
            image_low_res = interpolate(image[None], scale_factor=(1/6), mode='bilinear', align_corners=False , recompute_scale_factor=True)[0]
            # 裁剪+填充到192*256
            image_low_res = image_low_res[:, :, 32:-32]  # 低分辨率输入左右各裁32，torch.Size([3, 180, 256])
            image_low_res = pad(image_low_res[None], pad=[0, 0, 6, 6], mode="constant", value=0)[0]  # 低分辨率输入上下各+6，torch.Size([3, 192, 256])
            frames_low_res.append(image_low_res)  # 低分辨率输入序列
        #把这六个帧合为一个序列
        frames_low_res = torch.cat(frames_low_res, dim=0)#18,192,256
        seq_input = torch.reshape(frames_low_res,(1,frames_low_res.shape[0],frames_low_res.shape[1],frames_low_res.shape[2]))

        infer_data.append(seq_input)
    print('len:',len(infer_data))
    print('each element size:',infer_data[0].shape)

    return infer_data