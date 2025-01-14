import time

import tensorflow as tf
import torch
from torch.nn.functional import interpolate
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import matplotlib.pyplot as plt
import torchvision
import cv2
from PIL import Image
def getEXR(path):
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def init(path: str = './Bistro/train/train_sharp', number_of_frames: int = 1,
             overlapping_frames: int = 0, frame_format='exr') -> None:
    """
    Constructor method
    :param path: (str) Path to data
    :param number_of_frames: (int) Number of frames in one dataset element
    :param overlapping_frames: (int) Number of overlapping frames of two consecutive dataset elements
    :param frame_format: (str) Frame format to detect frames in path
    """
    # Call super constructor
    # super(REDS, self).__init__()
    # Save arguments
    # self.number_of_frames = number_of_frames
    # self.overlapping_frames = overlapping_frames
    # # Init previously loaded frames
    # self.previously_loaded_frames = None
    # # Init list to store all path to frames
    data_path = []
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
    return data_path
def getMask(img):#view_exr:c,h,w
#0. 准备模型和数据
    MASK_LOWER_BOUND = 0.35
    MASK_UPPER_BOUND = 0.5
    MASK_LOWER_VALUE = 0.2
    MASK_UPPER_VALUE = 0.98
    # print('---------------进入getmask--------------')
    model=tf.saved_model.load('./converted_model')
    model = model.signatures["serving_default"]

    # print('pil to np:',img_array.shape)
    img = img.transpose(1,2,0)
    img_array = tf.expand_dims(img,0)#b,h,w,c 1,720,1280,3
    # print('expand dims:',img_array.shape)


    heatmap = model(img_array)["output"].numpy()#numpy array b,h,w,c 1,720,1280,1 但180*320从此变成176*320
    heatmap=np.resize(heatmap,new_shape=(heatmap.shape[0],img_array.shape[1],img_array.shape[2],heatmap.shape[3]))

    # print('heatmap:',heatmap.shape)
    heatmap = tf.squeeze(heatmap)#h,w 720,1280 ##tensorflow也能对array做处理吗？
    p_mask = np.zeros_like(heatmap)

    #一阶函数：
    mask_lower = heatmap < MASK_LOWER_BOUND
    p_mask[mask_lower] = MASK_LOWER_VALUE
    mask_upper = heatmap > MASK_UPPER_BOUND
    p_mask[mask_upper] = MASK_UPPER_VALUE
    mask_middle = np.logical_and(heatmap >= MASK_LOWER_BOUND, heatmap <= MASK_UPPER_BOUND)
    m, b = np.linalg.pinv(np.array([[MASK_LOWER_BOUND, 1], [MASK_UPPER_BOUND, 1]])) @ np.array(
        [[MASK_LOWER_VALUE], [MASK_UPPER_VALUE]])
    p_mask[mask_middle] = m * heatmap[mask_middle] + b


    #阶梯函数：
    # p_mask[:, :] = np.where(heatmap[:, :] >= 0.5, 0.98,  p_mask[:, :])
    # p_mask[:, :] = np.where(np.logical_and(heatmap[:, :] < 0.5, heatmap[:, :] >= 0.35), 0.85, p_mask[:, :])
    # p_mask[:, :] = np.where(heatmap[:, :] < 0.35, 0.2, p_mask[:, :])



    mask = torch.from_numpy(p_mask >= np.random.uniform(low=0 , high=1 , size=(p_mask.shape[0],p_mask.shape[1]))).reshape(
        (p_mask.shape[0],p_mask.shape[1])
    )#7ms 从numpy到torch

    clock1 = time.time()
    # print("mask time:", clock1 - clock0)
    # print('mask:',mask.shape)
    return mask.float()
toPIL = torchvision.transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值

# mask_list=[]
# dataPath = init()
# for i in range(len(dataPath)):
#     # print(i)
#     exr_image = getEXR(dataPath[i][0])
#     image = torch.from_numpy(exr_image).permute(2, 0, 1)
#     image_low_res = interpolate(image[None], scale_factor=(1 / 4), mode='bilinear', align_corners=False,
#                                 recompute_scale_factor=True)[0]
#     mask = getMask(image_low_res.numpy())
#     # print(mask.shape)
#     mask_list.append(mask)
#     pic = toPIL(mask)
#     new_name = f'{int(i):04d}_0_mask.png'
#     pic.save(f'./save_mask/{new_name}')
#     print(dataPath[i][0])
# print(len(mask_list))
# print(dataPath)
# print(dataPath[0])
