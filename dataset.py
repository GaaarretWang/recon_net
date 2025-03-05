from typing import Tuple
# import getEyeMask
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.functional import interpolate
from torch.nn.functional import pad
import torchvision.transforms.functional as tf
from PIL import Image
import numpy as np
import os
from utils import save_image
from datetime import datetime
from model_wrapper import global_epoch
from masks import AngularCoord2ScreenCoord

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import torch.nn.functional as F
def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

class REDS(Dataset):
    """
    This class implements a video dataset for super resolution
    """

    def __init__(self, path: str = '../mixdataset/pictures/train', exclude_data: str = "", 
                 number_of_frames: int = 7) -> None:
        """
        Constructor method
        :param path: (str) Path to data
        :param batch_size: (int) Batch size
        :param number_of_frames: (int) Number of frames in one dataset element
        :param frame_format: (str) Frame format to detect frames in path
        """
        # Call super constructor
        super(REDS, self).__init__()
        # Save arguments
        self.number_of_frames = number_of_frames
        # self.batch_size = batch_size
        # Init previously loaded frames
        self.pre_scene = None
        # Init list to store all path to frames
        self.data_path = []
        # Get all objects in path an search for video folders
        for video in sorted(os.listdir(path=path)):
            if video == exclude_data:
                continue
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

                    # Add new list to data path in case of a new frame sequence
                    if frame_counter == 0:
                        self.data_path.append([])
                    # Add frame to last data path under list
                    self.data_path[-1].append(os.path.join(path, video, current_frame))
                    # Increment frame counter
                    frame_counter += 1
                    # Reset frame counter if number of frames in one element are reached
                    if frame_counter == number_of_frames:#!!!!!!
                        frame_counter = 0
                    # Increment frame index
                    frame_index += 1
        
    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length of the dataset
        """
        return len(self.data_path)

    @torch.no_grad()
    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Method returns one instance of the dataset for a given index
        :param item: (int) Index to get element
        :return: (Tuple[torch.Tensor, torch.Tensor]) Low res sequence, high res sequence, new video flag
        """
        # Check if current frame sequence is a new video sequence
        # if self.previously_loaded_frames is None or self.previously_loaded_frames[0].split('/')[-2] != \
        #         self.data_path[item][0].split('/')[-2]:
        #     new_video = True
        # else:
        #     new_video = False
        # # Set current data path to previously loaded frames
        # self.previously_loaded_frames = self.data_path[item]
        # # Load frames
        # frames_low_res = []
        # frames_label = []
        # for frame in self.data_path[item]:#遍历一个序列之内的所有帧
        #     # Load images as PIL image, and convert to tensor
        #     image = tf.to_tensor(Image.open(frame))
        #     # Normalize image to a mean of zero and a std of one
        #     # image = image.sub_(image.mean()).div_(image.std())
        #     # Downsampled frames
        #     image_low_res = interpolate(image[None], scale_factor=0.25, mode='bilinear', align_corners=False)[0]#720*1280 / 4=180*320
        #     # Crop normal image
        #     image = image[:, :, 128:-128]#720*1024
        #     image = pad(image[None], pad=[0, 0, 24, 24], mode="constant", value=0)[0]#上下填充24 768*1024
        #     # Crop low res masked image
        #     image_low_res = image_low_res[:, :, 32:-32]
        #     image_low_res = pad(image_low_res[None], pad=[0, 0, 6, 6], mode="constant", value=0)[0]
        #     # Add to list
        #     frames_low_res.append(image_low_res)
        #     # Add to list
        #     frames_label.append(image)
        # # Concatenate frames to tensor of shape (3 * number of frames, height (/ 4), width (/ 4))
        # frames_low_res = torch.cat(frames_low_res, dim=0)
        # frames_label = torch.cat(frames_label, dim=0)
        # return frames_low_res, frames_label, new_video


class REDSFovea(REDS):
    """
    Class implements the REDS dataset with a fovea sampled low resolution input sequence and a high resolution label
    """

    def __init__(self, path='../mixdataset/pictures/train', exclude_data: str = "",
                 number_of_frames: int = 7) -> None:
        """
        Constructor method
        :param path: (str) Path to data
        :param number_of_frames: (int) Number of frames in one dataset element
        :param frame_format: (str) Frame format to detect frames in path
        """
        # Call super constructor
        super(REDSFovea, self).__init__(path=path, exclude_data=exclude_data, number_of_frames=number_of_frames)
        # Init probability of mask
        print("4444")
        # print("model_wrapper.global_epoch = ")
        # print(global_epoch)
        # self.prevResult = None
        # self.path_save_plots = './saved_data/20230426-dataset_prev/'
        # self.mask = tf.to_tensor(Image.open('./Bistro/mask/mask.png'))
        # print('mask:',self.mask.shape)

    @torch.no_grad()
    def __getitem__(self, item: int) :
        """
        Get item method returns the fovea masked downsampled frame sequence, the high resolution sequence, and a bool
        if the new sequence is the start of a new video
        :param item: (int) Index to get element
        :return: (Tuple[torch.Tensor, torch.Tensor]) Low res fovea sampled sequence, high res sequence, new video flag
        """
        index = 0
        scene_name = os.path.basename(self.data_path[item][0].split(os.sep)[-2])
        for frame in self.data_path[item]:
            if index == 0:#img
                exr_image = getEXR(frame)#h w c
                label_image = torch.from_numpy(exr_image).permute(2,0,1)#c h w 1920*1080->320*180
            if index == 1:#albedo
                # albedo = torch.tensor(0)
                exr_image = getEXR(frame)#h w c
                projected_color = torch.from_numpy(exr_image).permute(2,0,1)#c h w 1920*1080->320*180
            if index == 2:#albedo
                # albedo = torch.tensor(0)
                exr_image = getEXR(frame)#h w c
                cur_color = torch.from_numpy(exr_image).permute(2,0,1)#c h w 1920*1080->320*180
            if index == 4:#sample_time_input
                exr_image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
                projected_sample_time = torch.from_numpy(exr_image).permute(2,0,1)[:1, :, :]
            if index == 5:#sample_time_input
                exr_image = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
                cur_sample_time = torch.from_numpy(exr_image).permute(2,0,1)[:1, :, :]
                # sample_time_input = torch.zeros(sample_time_input.shape)
            if index == 6:#img
                with open(frame, 'r') as infile:  
                    gaze_point = torch.from_numpy(np.loadtxt(infile))  # data = array([0.5 , 1.75])              index = index + 1
            index = index + 1
        # gaze_point = AngularCoord2ScreenCoord(gaze_point, [label_image.shape[2], label_image.shape[1]])
        # label_albedo *= mask.float()  
        projected_sample_time = torch.where(projected_sample_time < 90, (90 - projected_sample_time) / 90.0, torch.tensor(0.0))  
        return projected_color, cur_color, projected_sample_time, cur_sample_time, label_image, gaze_point, scene_name


class REDSParallel(Dataset):
    """
    Wraps the REDS dataset for multi gpu usage. The number of gpus must match the batch size. One batch on one GPU
    """

    def __init__(self, path: str = '/home/creich/REDS/train/train_sharp', number_of_frames: int = 6,
                 overlapping_frames: int = 2, frame_format='png', number_of_gpus: int = 2) -> None:
        """
        Constructor method
        :param path: (str) Path to data
        :param number_of_frames: (int) Number of frames in one dataset element
        :param overlapping_frames: (int) Number of overlapping frames of two consecutive dataset elements
        :param frame_format: (str) Frame format to detect frames in path
        """
        # Init for each gpu one dataset
        self.datasets = [REDS(path=path, number_of_frames=number_of_frames, overlapping_frames=overlapping_frames) for _ in range(number_of_gpus)]
        # Save parameters
        self.number_of_gpus = number_of_gpus

    def __len__(self) -> int:
        """
        Method to get the length of the dataset
        :return: (int) Length
        """
        return len(self.datasets[0])

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Get item method returns the downsampled frame sequence, the high resolution sequence, and a bool
        if the new sequence is the start of a new video
        :param item: (int) Index to get element
        :return: (Tuple[torch.Tensor, torch.Tensor]) Low res fovea sampled sequence, high res sequence, new video flag
        """
        # Get gpu index corresponding to item value
        gpu_index = item % self.number_of_gpus
        # Calc offset for item
        item = (item // self.number_of_gpus) + ((len(self) // self.number_of_gpus) * gpu_index)
        return self.datasets[gpu_index][item]


class REDSFoveaParallel(Dataset):
    """
    Wraps the REDS fovea dataset for multi gpu usage. The number of gpus must match the batch size. One batch on one GPU
    """

    def __init__(self, path: str = '/home/creich/REDS/train/train_sharp', number_of_frames: int = 6,
                 overlapping_frames: int = 0, frame_format='exr', number_of_gpus: int = 2) -> None:
        """
        Constructor method
        :param path: (str) Path to data
        :param number_of_frames: (int) Number of frames in one dataset element
        :param overlapping_frames: (int) Number of overlapping frames of two consecutive dataset elements
        :param frame_format: (str) Frame format to detect frames in path
        """
        # Init for each gpu one dataset
        self.datasets = [REDSFovea(path=path, number_of_frames=number_of_frames, overlapping_frames=overlapping_frames,
                                   frame_format=frame_format) for _ in range(number_of_gpus)]
        # Save parameters
        self.number_of_gpus = number_of_gpus

    def __len__(self) -> int:
        """
        Method to get the length of the dataset
        :return: (int) Length
        """
        return len(self.datasets[0])

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Get item method returns the fovea masked downsampled frame sequence, the high resolution sequence, and a bool
        if the new sequence is the start of a new video
        :param item: (int) Index to get element
        :return: (Tuple[torch.Tensor, torch.Tensor]) Low res fovea sampled sequence, high res sequence, new video flag
        """
        # Get gpu index corresponding to item value
        gpu_index = item % self.number_of_gpus
        # Calc offset for item
        item = (item // self.number_of_gpus) + ((len(self) // self.number_of_gpus) * gpu_index)
        return self.datasets[gpu_index][item]


def reds_parallel_collate_fn(batch: Tuple[torch.Tensor, torch.Tensor, bool]) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """
    Collate function for parallel dataset to manage new_video flag
    :param batch: (Tuple[torch.Tensor, torch.Tensor, bool]) Batch
    :return: (Tuple[torch.Tensor, torch.Tensor, bool]) Stacked input & label and new_video flag
    """
    input = torch.stack([batch[index][0] for index in range(len(batch))], dim=0)
    label = torch.stack([batch[index][1] for index in range(len(batch))], dim=0)
    new_video = sum([batch[index][2] for index in range(len(batch))]) != 0
    return input, label, new_video


class PseudoDataset(Dataset):
    """
    This class implements a pseudo dataset to test the implemented architecture
    """

    def __init__(self, length: int = 100) -> None:
        """
        Constructor method
        :param length: (int) Pseudo dataset length
        """
        # Call super constructor
        super(PseudoDataset, self).__init__()
        # Save length parameter
        self.length = length

    def __len__(self) -> int:
        """
        Method to get length of the dataset
        :return: (int) Length
        """
        return self.length

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method returns a tensor of the shape (rgb * 16 frames, height, width) and the corresponding high res. label
        :param item: (int) Index
        :return: (torch.Tensor) Pseudo image sequence and corresponding label
        """
        if item >= len(self):
            raise IndexError
        return torch.ones([3 * 16, 64, 64], dtype=torch.float), torch.ones([3 * 16, 256, 256], dtype=torch.float)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = DataLoader(REDSFoveaParallel(), shuffle=False, num_workers=1, batch_size=2,
                         collate_fn=reds_parallel_collate_fn)
    print(dataset.dataset.__len__())
    counter = 0
    for input, label, new_video in dataset:
        print(torch.sum((input != 0).float()) / input.numel())
        print(torch.sum((input != 0).float()) / label.numel())
        exit(22)
