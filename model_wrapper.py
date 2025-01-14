
from itertools import islice
from typing import Callable, Union, Tuple, List
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from utils import upsample_zero_2d
import torch.autograd
import torchvision
import os
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torch.nn.functional import interpolate
import lossfunction
import misc
from utils import save_image
import pytorch_ssim
import cv2
import random
import math
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import draw_bounding_boxes
from numba import cuda
from numba import njit

from warpFunctions import pre_process, warp
from masks import get_center_mask, AngularCoord2ScreenCoord
global_epoch = 0
                  
class VGG(nn.Module):
    def __init__(self, conv_index):
        super(VGG, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).to("cuda:0").features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        vgg_sr = self.vgg(sr)
        with torch.no_grad():
            vgg_hr = self.vgg(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss

class ModelWrapper(object):
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    """
    This class wraps all modules and implements train, validation, test and inference methods
    """
    def __init__(self,generator_network: Union[nn.Module, nn.DataParallel],
                #  vgg_19: Union[nn.Module, nn.DataParallel],
                 L1Loss: Union[nn.Module, nn.DataParallel],
                 lpips: Union[nn.Module, nn.DataParallel],
                 generator_network_optimizer: torch.optim.Optimizer,
                 scheduler : torch.optim.lr_scheduler,
                 details : str,
                 batch_size : int,
                 training_dataloader: DataLoader,
                 validation_dataloader: DataLoader,
                 validation_dataloader_1: DataLoader,
                 test_dataloader: DataLoader,
                 width : int, 
                 height : int,
                 perceptual_loss: nn.Module = lossfunction.PerceptualLoss(),
                 generator_loss: nn.Module = lossfunction.NonSaturatingLogisticGeneratorLoss(),
                 device='cuda',
                 save_data_path: str = 'saved_data') -> None:
        """
        Constructor method
        :param generator_network: (nn.Module) Generator models
        :param discriminator_network: (nn.Module) Discriminator model
        :param fft_discriminator_network: (nn.Module) FFT discriminator model
        :param vgg_19: (nn.Module) Pre-trained VGG19 network
        :param pwc_net: (nn.Module) PWC-Net for optical flow estimation
        :param resample: (nn.Module) Resampling module
        :param generator_network_optimizer: (torch.optim.Optimizer) Generator optimizer module
        :param discriminator_network_optimizer: (torch.optim.Optimizer) Discriminator optimizer
        :param fft_discriminator_network_optimizer: (torch.optim.Optimizer) FFT discriminator model
        :param training_dataloader: (DataLoader) Training dataloader including the training dataset
        :param validation_dataloader: (DataLoader) Validation dataloader including the validation dataset
        :param test_dataloader: (DataLoader) Test dataloader including the test dataset
        :param loss_function: (nn.Module) Main supervised loss function
        :param perceptual_loss: (nn.Module) Perceptual loss function which takes two lists of tensors as input
        :param flow_loss: (nn.Module) Flow loss function
        :param generator_loss: (nn.Module) Adversarial generator loss function
        :param discriminator_loss: (nn.Module) Adversarial discriminator loss function
        :param device: (str) Device to be utilized (cpu not available if deformable convolutions are utilized)
        :param save_data_path: (str) Path to store logs, models and plots
        """
        # Save arguments
        self.generator_network = generator_network
        # self.vgg_19 = vgg_19
        # self.vgg22 = VGG('22')
        self.L1Loss = L1Loss
        self.lpips = lpips
        self.generator_network_optimizer = generator_network_optimizer
        self.scheduler = scheduler

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.validation_dataloader_1 = validation_dataloader_1
        self.test_dataloader = test_dataloader

        self.perceptual_loss = perceptual_loss
        self.generator_loss = generator_loss
        self.batch_size = batch_size

        self.device = device
        self.save_data_path = save_data_path
        # Make directories to save logs, plots and models during training
        # Not compatible with windows!!!
        time_and_date = str(datetime.now()).replace(' ', '_').replace(':', '.')
        save_data_path = os.path.join(save_data_path, details + time_and_date)
        # time_and_date = str(datetime.now())
        self.path_save_models = os.path.join(save_data_path, 'models')
        if not os.path.exists(self.path_save_models):
            os.makedirs(self.path_save_models)
        self.path_save_plots = os.path.join(save_data_path, 'plots')
        if not os.path.exists(self.path_save_plots):
            os.makedirs(self.path_save_plots)
        self.path_save_metrics = os.path.join(save_data_path, 'metrics')
        if not os.path.exists(self.path_save_metrics):
            os.makedirs(self.path_save_metrics)

        self.pre_input = []
        self.mvs = None
        self.width = width
        self.height = height

        print("5555")

    def downsample_with_max_weight(self, rgb_image, weight_map, n):  
        # 假设 rgb_image 和 weight_map 的形状都是 (B, C, H, W)  
        B, C, H, W = rgb_image.size()  

        # 计算低分辨率尺寸  
        low_res_H = H // n  
        low_res_W = W // n  

        # 初始化低分辨率图像  
        low_res_rgb = torch.zeros((B, C, low_res_H, low_res_W), device=rgb_image.device)  

        # 遍历每个样本  
        for b in range(B):  
            # 遍历每个 n*n 区域  
            for i in range(low_res_H):  
                for j in range(low_res_W):  
                    # 确定当前区域的位置  
                    h_start, h_end = i * n, (i + 1) * n  
                    w_start, w_end = j * n, (j + 1) * n  
                    
                    # 提取区域的权重图和 RGB 图像  
                    region_weights = weight_map[b, :, h_start:h_end, w_start:w_end]  
                    region_rgb = rgb_image[b, :, h_start:h_end, w_start:w_end]  

                    # 找到最高权重像素的位置  
                    max_weight_idx = torch.argmax(region_weights.reshape(-1))  
                    max_weight_pos = divmod(max_weight_idx.item(), n * n)  
                    h_offset = max_weight_pos[1] // n
                    w_offset = max_weight_pos[1] % n
                    # print(region_weights)
                    # print(max_weight_pos)
                    # print(h_offset)
                    # print(w_offset)
                    # print(region_weights[:, h_offset, w_offset])
                    # 获取最高权重像素的 RGB 值  
                    low_res_rgb[b, :, i, j] = region_rgb[:, h_offset, w_offset]  
        
        return low_res_rgb  


    def train(self, epochs: int = 1, save_models_after_n_epochs: int = 1, validate_after_n_epochs: int = 3,
              w_perceptual: float = 1.0, w_flow: float = 2.0,
              plot_after_n_iterations: int = 500) -> None:#迭代次数=epoch * (2500/2)=，可以每625次保存,一个epoch保存两次
        """
        Train method
        Note: GPU memory issues if all losses are computed at one. Solution: Calc losses independently. Drawback:
        Multiple forward passes are needed -> Slower training. Additionally gradients are not as smooth.
        :param epochs: (int) Number of epochs to perform
        :param save_models_after_n_epochs: (int) Epochs after models and optimizers gets saved
        :param validate_after_n_epochs: (int) Perform validation after a given number of epochs
        :param w_supervised_loss: (float) Weight factor for the supervised loss
        :param w_adversarial: (float) Weight factor for adversarial generator loss
        :param w_fft_adversarial: (float) Weight factor for fft adversarial generator loss
        :param w_perceptual: (float) Weight factor for perceptual loss
        :param w_flow: (float) Weight factor for flow loss
        :param inference_plot_after_n_iterations: (int) Make training plot after a given number of iterations
        """

        self.setup_seed(9910)
        # Model into training mode
        # self.generator_network.module.initialize_weights()
        self.generator_network.train()

        # PWC-Net into eval mode
        # self.depth_network.eval()
        # Vgg into eval mode
        # self.vgg_19.eval()
        # self.vgg22.eval()
        self.lpips.eval()
        # Models to device
        self.generator_network.to(self.device)
        # self.depth_network.to(self.device)
        # self.vgg_19.to(self.device)
        # self.vgg22.to(self.device)
        # Init progress bar
        self.progress_bar = tqdm(total=epochs * (len(self.training_dataloader.dataset)))
        '''
        这行代码使用了tqdm库，用于显示训练的进度条。其中total表示总共需要迭代的次数，即epochs * len(self.training_dataloader.dataset)；
        epochs表示训练的轮数，len(self.training_dataloader.dataset)表示训练集的数据量。如80张exr，6个为1个data，共15个
        在每次迭代时，tqdm会自动更新进度条的进度并显示当前训练的状态。
        '''
        self.notnan_generatornetwork = self.generator_network
        self.notnan_optimizer = self.generator_network_optimizer
        crop_size = 1024
        # Main loop
        for epoch in range(epochs):
            global global_epoch
            global_epoch = epoch
            self.all_one = torch.ones([1, 1, crop_size, crop_size]).to('cuda')
            # prevResult = None
            # pre_ref = None
            for index_sequence, batch in enumerate(self.training_dataloader):  # 遍历每个序列
                '''
                input:3*4+2,192 256
                label:3*1 192 256
                '''
                # input, label, label_mv = batch  # input：img albedo normal depth mv,label:img
                projected_color, cur_color, projected_sample_time, cur_sample_time, label_image, gaze_point = batch
                random_x = random.randint(0, self.width - crop_size)
                random_y = random.randint(0, self.height - crop_size)
                projected_color = projected_color.to(self.device)[:, :, random_y:random_y+crop_size, random_x:random_x+crop_size]
                cur_color = cur_color.to(self.device)[:, :, random_y:random_y+crop_size, random_x:random_x+crop_size]
                projected_sample_time = projected_sample_time.to(self.device)[:, :, random_y:random_y+crop_size, random_x:random_x+crop_size]
                cur_sample_time = cur_sample_time.to(self.device)[:, :, random_y:random_y+crop_size, random_x:random_x+crop_size]
                label_image = label_image.to(self.device)[:, :, random_y:random_y+crop_size, random_x:random_x+crop_size]
                gaze_point = gaze_point.to(self.device)

                # Reset gradients of networks
                self.generator_network.zero_grad()
                # self.vgg_19.zero_grad()
                # self.vgg22.zero_grad()
                # Update progress bar
                self.progress_bar.update(n=gaze_point.shape[0])

                # grid_sub = grid - grid_norm
                # grid_sub[:, 0:1, :, :] *= self.width // 2
                # grid_sub[:, 1:2, :, :] *= self.height // 2
                ############# Supervised training (+ perceptrual training) #############
                # Make prediction
                all_fovea_masks = get_center_mask(gaze_point, [self.width, self.height])[:, :, random_y:random_y+crop_size, random_x:random_x+crop_size]
                # all_fovea_masks = torch.ones(all_fovea_masks.shape).to('cuda')  
                prediction_image = self.generator_network(projected_color, cur_color, projected_sample_time, cur_sample_time)  # b c h w=1,18,h,w  1,6,h,w#generator要改channel

                loss_spatial = self.L1Loss(prediction_image, label_image) / 6
                loss_ssim_fovea = torch.mean(all_fovea_masks) - pytorch_ssim.ssim(prediction_image, label_image, all_fovea_masks)
                loss_ssim = 1 - pytorch_ssim.ssim(prediction_image, label_image, self.all_one)
                loss_lpips = self.lpips(prediction_image, label_image, all_fovea_masks)
                
                downsampled_prediction_image = F.avg_pool2d(prediction_image, kernel_size=8, stride=8)  
                downsampled_label_image = F.avg_pool2d(label_image, kernel_size=8, stride=8)  

                loss_shadow = torch.mean((downsampled_prediction_image + 0.01) / (downsampled_label_image + 0.01) - 1) ** 2
                # if index_sequence == 0:
                #     loss_temporal = self.L1Loss(
                #         (prediction - prediction),
                #         (label_image - label_image),
                #         fovea_mask
                #     )
                # else:
                #     loss_temporal = self.L1Loss(
                #         (prediction - prevResult),
                #         (label_image - pre_ref),
                #         fovea_mask
                #     )
                # Calc gradients
                # prevResult = prediction.detach()  # 1 3 768 1024
                # pre_ref = label_image.detach()
                all_loss = (loss_spatial  + loss_ssim * 2 + loss_lpips + loss_shadow)#L1+VGG
                all_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator_network.parameters(), max_norm=0.16)
                # Optimize generator
                self.generator_network_optimizer.step()
                # self.DGaze_ET_optimizer.step()
                # for name, param in self.DGaze_ET_model.named_parameters():  
                #     print(f"Parameter name: {name}")  
                #     print(f"Parameter value: {param}")  
                #     print(f"Parameter size: {param.size()}\n")  
                #     break
                # Reset gradients of generator network
                # self.generator_network.zero_grad()
                # if torch.isnan(all_loss):
                #     prevResult = label_img
                #     self.generator_network = copy.deepcopy(self.notnan_generatornetwork)
                #     self.generator_network_optimizer = copy.deepcopy(self.notnan_optimizer)
                # elif index_sequence % 20 == 0:
                #     self.notnan_generatornetwork = copy.deepcopy(self.generator_network)
                #     self.notnan_optimizer = copy.deepcopy(self.generator_network_optimizer)
                
                # Update progress bar
                self.progress_bar.set_description(
                    'L1 Loss={:.3f}, ssim Loss={:.3f}, ssim Loss fovea={:.3f}, shaodw_loss={:.3f}, all Loss={:.3f}'
                    # 'L1 Loss={:.3f}, ssim Loss={:.3f}, T Loss = {:.3f} , all Loss={:.3f}'
                    .format(loss_spatial.item(),
                            loss_ssim.item(),
                            loss_ssim_fovea.item(),
                            loss_shadow.item(),
                            all_loss.item()))

                # Plot training prediction
                if (self.progress_bar.n) % (plot_after_n_iterations) == 0:
                    # Normalize images batch wise to range of [0, 1]
                    '''prediction_batched = misc.normalize_0_1_batch(prediction_batched)
                    label_batched = misc.normalize_0_1_batch(label_batched)'''
                    # Make plots
                    save_image(
                        prediction_image,
                        fp=os.path.join(self.path_save_plots,
                                        'prediction_train_{}.exr'.format(self.progress_bar.n)),
                        format='exr',
                        nrow=self.validation_dataloader.dataset.number_of_frames)

            self.scheduler.step()
            # print(self.generator_network_optimizer.state_dict()['param_groups'][0]['lr'])
            # Save models and optimizer
            if (epoch + 1) % save_models_after_n_epochs == 0:
                # Save models
                torch.save(self.generator_network,
                           os.path.join(self.path_save_models, 'generator_network_model_{}.pt'.format(epoch)))

                # # Save optimizers
                # torch.save(self.generator_network_optimizer,
                #            os.path.join(self.path_save_models, 'generator_network_optimizer_{}.pt'.format(epoch)))

            # print('train_over')
            if (epoch + 1) % validate_after_n_epochs == 0:
                # Validation
                self.progress_bar.set_description('Validate...')
                self.validate(epoch = epoch)
                self.validate_1(epoch = epoch)
            # print(self.generator_network_optimizer.param_groups[0]['lr'])
        # Close progress bar
        self.progress_bar.close()

    @torch.no_grad()
    def validate(self, epoch = 0, plot_after_n_iterations = 500,
                 validation_metrics: Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]
                 = (nn.L1Loss(reduction='mean'), nn.MSELoss(reduction='mean'), misc.psnr, misc.ssim)) -> None:
        """
        Validation method which produces validation metrics and plots
        :param validation_metrics: (Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]) Tuple
        of callable validation metric to be computed
        :param sequences_to_plot: (Tuple[int, ...]) Tuple of validation dataset indexes to be plotted
        :param reset_recurrent_tensor_after_each_sequnece: (bool) Resets recurrent tensor in case of a sequence if true
        """
        # Generator model to device
        self.setup_seed(9910)
        self.generator_network.to(self.device)
        self.generator_network.eval()
        # Main loop
        psnr_sum = 0
        ssim_sum = 0
        cropped_psnr_sum = 0
        cropped_ssim_sum = 0
        num = 0
        crop_size = 300
        pad_size = int(crop_size / 2)
        pre_warp_color = None
        grid = None
        grid_norm = None
        sample_time = None
        warped_id = None
        num_10000 = (torch.full((1, 1, self.height, self.width), -10000.0).to('cuda'))
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
        psnr = PeakSignalNoiseRatio().to('cuda')
        frame_index = 0

        # self.ssim_sub = []
        # with open('./ssim_tmp/ssim_0.txt', "r") as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         self.ssim_sub.append(float(line))

        for index_sequence, batch in enumerate(self.validation_dataloader):  # 遍历每个序列
            projected_color, cur_color, projected_sample_time, cur_sample_time, label_image, gaze_point = batch
            projected_color = projected_color.to(self.device)
            cur_color = cur_color.to(self.device)
            projected_sample_time = projected_sample_time.to(self.device)
            cur_sample_time = cur_sample_time.to(self.device)
            label_image = label_image.to(self.device)
            gaze_point = gaze_point.to(self.device)
            
            # b, c, h, w = label_image.shape
            # x = torch.linspace(-1, 1, steps=w + 1)[:-1] + 1 / w
            # y = torch.linspace(-1, 1, steps=h + 1)[:-1] + 1 / h
            # grid_y, grid_x = torch.meshgrid(y, x)
            # grid_norm = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).to(self.device)
                                        
            # grid_sub = grid - grid_norm
            # grid_sub[:, 0:1, :, :] *= self.width // 2
            # grid_sub[:, 1:2, :, :] *= self.height // 2

            # Make prediction
            all_fovea_masks = get_center_mask(gaze_point, [self.width, self.height])  
            prediction_image = self.generator_network(projected_color, cur_color, projected_sample_time, cur_sample_time)  # b c h w=1,18,h,w  1,6,h,w#generator要改channel
            # cur_gaze_coord = gaze_labels[0, 0:2]
            # cur_gaze_coord = [max(0, min(int(cur_gaze_coord[0]), 1919)), max(0, min(int(cur_gaze_coord[1]), 1079))]
            # pad_prediction = F.pad(prediction, (pad_size, pad_size, pad_size, pad_size))
            # pad_label_img = F.pad(label_img, (pad_size, pad_size, pad_size, pad_size))
            # cropped_prediction = pad_prediction[:, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
            # cropped_label_img = pad_label_img[:, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
            ssim_sum += ssim(prediction_image, label_image)
            psnr_sum += psnr(prediction_image, label_image)
            pad_prediction = F.pad(prediction_image, (pad_size, pad_size, pad_size, pad_size))
            pad_label_img = F.pad(label_image, (pad_size, pad_size, pad_size, pad_size))
            for i in range(gaze_point.shape[0]):  
                cur_gaze_coord = gaze_point[i, 0:2]
                cur_gaze_coord = [max(0, min(int(cur_gaze_coord[0]), self.width - 1)), max(0, min(int(cur_gaze_coord[1]), self.height - 1))]
                cropped_predictions = pad_prediction[i:i+1, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
                cropped_label_imgs = pad_label_img[i:i+1, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
                cropped_ssim_sum += ssim(cropped_predictions, cropped_label_imgs)
                cropped_psnr_sum += psnr(cropped_predictions, cropped_label_imgs)

            # cropped_psnr_sum += psnr(cropped_prediction, cropped_label_img)
            # # print(psnr(cropped_prediction, cropped_label_img))
            # cropped_ssim_sum += ssim(cropped_prediction, cropped_label_img)
            # print(ssim(cropped_prediction, cropped_label_img))
            # print(f"Index {index_sequence}, CUR_CROPPED_PSNR: {psnr(cropped_prediction, cropped_label_img):.4f}, CUR_CROPPED_SSIM: {ssim(cropped_prediction, cropped_label_img):.4f}")
            # print(f"{index_sequence} {psnr(cropped_prediction, cropped_label_img):.4f} {ssim(cropped_prediction, cropped_label_img):.4f}")
            # with open(os.path.join(self.path_save_models, "valid_ssim.txt"), "a") as f:
            #     f.write(f"{ssim(cropped_prediction, cropped_label_img):.4f}\n")

            # print(f"{ssim(cropped_prediction, cropped_label_img):.4f}")
            num += gaze_point.shape[0]          

            # prediction_gaze_position = torch.clamp(prediction[0] * 255, min=0, max=255).to(torch.uint8).cpu()
            # for thickness in range(0, 3):
            #     boxes = torch.tensor([[cur_gaze_coord[0]-pad_size-thickness, cur_gaze_coord[1]-pad_size-thickness, cur_gaze_coord[0]+pad_size+thickness, cur_gaze_coord[1]+pad_size+thickness]])  # 形状为(N, 4)，这里N=1
            #     prediction_gaze_position = draw_bounding_boxes(prediction_gaze_position, boxes, colors=[(255, 0, 0)])
            #     boxes = torch.tensor([[cur_gaze_coord[0]-pad_size+thickness, cur_gaze_coord[1]-pad_size+thickness, cur_gaze_coord[0]+pad_size-thickness, cur_gaze_coord[1]+pad_size-thickness]])  # 形状为(N, 4)，这里N=1
            #     prediction_gaze_position = draw_bounding_boxes(prediction_gaze_position, boxes, colors=[(255, 0, 0)])
            # torch.cuda.synchronize()

            if index_sequence % plot_after_n_iterations == 0:
                # dummy_input =(grid_sub, albedo, normal, input_image, sample_time_input)
                # torch.onnx.export(self.generator_network.module,
                #                 dummy_input,
                #                 'generator_network_2.onnx',
                #                 verbose=True,
                #                 opset_version=11,
                #                 operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

                # save_image(
                #     prediction_gaze_position.to(torch.float32) / 255,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                save_image(
                    prediction_image,
                    fp=os.path.join(self.path_save_plots,
                                    'prediction_warp_{}_{}.exr'.format(epoch,index_sequence)),
                    format='exr',
                    nrow=self.validation_dataloader.dataset.number_of_frames)                  
                save_image(
                    label_image,
                    fp=os.path.join(self.path_save_plots,
                                    'label_image_{}_{}.exr'.format(epoch,index_sequence)),
                    format='exr',
                    nrow=self.validation_dataloader.dataset.number_of_frames)                  
                # save_image(
                #     sample_time_input,
                #     fp=os.path.join(self.path_save_plots,
                #                     'sample_time_input_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)                  
                # save_image(
                #     out_color,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_warp_process_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     out_color1,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_warp_process1_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     out_color2,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_warp_process2_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.clamp(torch.reciprocal(sample_time+1) * 2, 0, 1),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_time_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     F.interpolate(label_img, scale_factor=(1 / 4), mode='bilinear', align_corners=False),
                #     fp=os.path.join(self.path_save_plots,
                #                     'label_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.cat((num_1, grid_sub), dim=1),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_grid_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     mask_albedo,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_mask_albedo_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     mask_normal,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_mask_normal_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     pre_warp_albedo,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_pre_warp_albedo_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     pre_warp_normal,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_pre_warped_normal_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.abs(pre_warp_albedo - mask_albedo),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_sub_albedo_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.abs(pre_warp_normal - mask_normal),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_sub_normal_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)

        print(f"Epoch {epoch}, PSNR: {psnr_sum / num:.4f}, SSIM: {ssim_sum / num:.4f}")
        print(f"Epoch {epoch}, CUR_CROPPED_PSNR: {cropped_psnr_sum / num:.4f}, CUR_CROPPED_SSIM: {cropped_ssim_sum / num:.4f}")
        with open(os.path.join(self.path_save_models, "valid_data.txt"), "a") as f:
            f.write(f"Epoch {epoch}, PSNR: {psnr_sum / num:.4f}, SSIM: {ssim_sum / num:.4f}\n")
            f.write(f"Epoch {epoch}, CUR_CROPPED_PSNR: {cropped_psnr_sum / num:.4f}, CUR_CROPPED_SSIM: {cropped_ssim_sum / num:.4f}\n")

    @torch.no_grad()
    def validate_1(self, epoch = 0, plot_after_n_iterations = 500,
                 validation_metrics: Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]
                 = (nn.L1Loss(reduction='mean'), nn.MSELoss(reduction='mean'), misc.psnr, misc.ssim)) -> None:
        """
        Validation method which produces validation metrics and plots
        :param validation_metrics: (Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]) Tuple
        of callable validation metric to be computed
        :param sequences_to_plot: (Tuple[int, ...]) Tuple of validation dataset indexes to be plotted
        :param reset_recurrent_tensor_after_each_sequnece: (bool) Resets recurrent tensor in case of a sequence if true
        """
        # Generator model to device
        self.setup_seed(9910)
        self.generator_network.to(self.device)
        self.generator_network.eval()
        # Main loop
        psnr_sum = 0
        ssim_sum = 0
        cropped_psnr_sum = 0
        cropped_ssim_sum = 0
        num = 0
        crop_size = 300
        pad_size = int(crop_size / 2)
        pre_warp_color = None
        grid = None
        grid_norm = None
        sample_time = None
        warped_id = None
        num_10000 = (torch.full((1, 1, self.height, self.width), -10000.0).to('cuda'))
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
        psnr = PeakSignalNoiseRatio().to('cuda')
        frame_index = 0

        # self.ssim_sub = []
        # with open('./ssim_tmp/ssim_0.txt', "r") as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         self.ssim_sub.append(float(line))

        for index_sequence, batch in enumerate(self.validation_dataloader_1):  # 遍历每个序列
            projected_color, cur_color, projected_sample_time, cur_sample_time, label_image, gaze_point = batch
            projected_color = projected_color.to(self.device)
            cur_color = cur_color.to(self.device)
            projected_sample_time = projected_sample_time.to(self.device)
            cur_sample_time = cur_sample_time.to(self.device)
            label_image = label_image.to(self.device)
            gaze_point = gaze_point.to(self.device)
            # down_img = self.downsample_with_max_weight(input_image, sample_time_input, 16)      
            # b, c, h, w = label_image.shape
            # x = torch.linspace(-1, 1, steps=w + 1)[:-1] + 1 / w
            # y = torch.linspace(-1, 1, steps=h + 1)[:-1] + 1 / h
            # grid_y, grid_x = torch.meshgrid(y, x)
            # grid_norm = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).to(self.device)
                                        
            # grid_sub = grid - grid_norm
            # grid_sub[:, 0:1, :, :] *= self.width // 2
            # grid_sub[:, 1:2, :, :] *= self.height // 2

            # Make prediction
            all_fovea_masks = get_center_mask(gaze_point, [self.width, self.height])  
            prediction_image = self.generator_network(projected_color, cur_color, projected_sample_time, cur_sample_time)  # b c h w=1,18,h,w  1,6,h,w#generator要改channel
            # cur_gaze_coord = gaze_labels[0, 0:2]
            # cur_gaze_coord = [max(0, min(int(cur_gaze_coord[0]), 1919)), max(0, min(int(cur_gaze_coord[1]), 1079))]
            # pad_prediction = F.pad(prediction, (pad_size, pad_size, pad_size, pad_size))
            # pad_label_img = F.pad(label_img, (pad_size, pad_size, pad_size, pad_size))
            # cropped_prediction = pad_prediction[:, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
            # cropped_label_img = pad_label_img[:, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
            ssim_sum += ssim(prediction_image, label_image)
            psnr_sum += psnr(prediction_image, label_image)
            pad_prediction = F.pad(prediction_image, (pad_size, pad_size, pad_size, pad_size))
            pad_label_img = F.pad(label_image, (pad_size, pad_size, pad_size, pad_size))
            for i in range(gaze_point.shape[0]):  
                cur_gaze_coord = gaze_point[i, 0:2]
                cur_gaze_coord = [max(0, min(int(cur_gaze_coord[0]), self.width - 1)), max(0, min(int(cur_gaze_coord[1]), self.height - 1))]
                cropped_predictions = pad_prediction[i:i+1, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
                cropped_label_imgs = pad_label_img[i:i+1, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
                cropped_ssim_sum += ssim(cropped_predictions, cropped_label_imgs)
                cropped_psnr_sum += psnr(cropped_predictions, cropped_label_imgs)

            # cropped_psnr_sum += psnr(cropped_prediction, cropped_label_img)
            # # print(psnr(cropped_prediction, cropped_label_img))
            # cropped_ssim_sum += ssim(cropped_prediction, cropped_label_img)
            # print(ssim(cropped_prediction, cropped_label_img))
            # print(f"Index {index_sequence}, CUR_CROPPED_PSNR: {psnr(cropped_prediction, cropped_label_img):.4f}, CUR_CROPPED_SSIM: {ssim(cropped_prediction, cropped_label_img):.4f}")
            # print(f"{index_sequence} {psnr(cropped_prediction, cropped_label_img):.4f} {ssim(cropped_prediction, cropped_label_img):.4f}")
            # with open(os.path.join(self.path_save_models, "valid_ssim.txt"), "a") as f:
            #     f.write(f"{ssim(cropped_prediction, cropped_label_img):.4f}\n")

            # print(f"{ssim(cropped_prediction, cropped_label_img):.4f}")
            num += gaze_point.shape[0]          

            # prediction_gaze_position = torch.clamp(prediction[0] * 255, min=0, max=255).to(torch.uint8).cpu()
            # for thickness in range(0, 3):
            #     boxes = torch.tensor([[cur_gaze_coord[0]-pad_size-thickness, cur_gaze_coord[1]-pad_size-thickness, cur_gaze_coord[0]+pad_size+thickness, cur_gaze_coord[1]+pad_size+thickness]])  # 形状为(N, 4)，这里N=1
            #     prediction_gaze_position = draw_bounding_boxes(prediction_gaze_position, boxes, colors=[(255, 0, 0)])
            #     boxes = torch.tensor([[cur_gaze_coord[0]-pad_size+thickness, cur_gaze_coord[1]-pad_size+thickness, cur_gaze_coord[0]+pad_size-thickness, cur_gaze_coord[1]+pad_size-thickness]])  # 形状为(N, 4)，这里N=1
            #     prediction_gaze_position = draw_bounding_boxes(prediction_gaze_position, boxes, colors=[(255, 0, 0)])
            # torch.cuda.synchronize()

            if index_sequence % plot_after_n_iterations == 0:
                # dummy_input =(grid_sub, albedo, normal, input_image, sample_time_input)
                # torch.onnx.export(self.generator_network.module,
                #                 dummy_input,
                #                 'generator_network_2.onnx',
                #                 verbose=True,
                #                 opset_version=11,
                #                 operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

                # save_image(
                #     prediction_gaze_position.to(torch.float32) / 255,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                save_image(
                    prediction_image,
                    fp=os.path.join(self.path_save_plots,
                                    'prediction_warp_{}_{}.exr'.format(epoch,index_sequence)),
                    format='exr',
                    nrow=self.validation_dataloader.dataset.number_of_frames)                  
                save_image(
                    label_image,
                    fp=os.path.join(self.path_save_plots,
                                    'label_image_{}_{}.exr'.format(epoch,index_sequence)),
                    format='exr',
                    nrow=self.validation_dataloader.dataset.number_of_frames)                  
                # save_image(
                #     out_color,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_warp_process_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     out_color1,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_warp_process1_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     out_color2,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_warp_process2_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.clamp(torch.reciprocal(sample_time+1) * 2, 0, 1),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_time_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     F.interpolate(label_img, scale_factor=(1 / 4), mode='bilinear', align_corners=False),
                #     fp=os.path.join(self.path_save_plots,
                #                     'label_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.cat((num_1, grid_sub), dim=1),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_grid_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     mask_albedo,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_mask_albedo_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     mask_normal,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_mask_normal_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     pre_warp_albedo,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_pre_warp_albedo_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     pre_warp_normal,
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_pre_warped_normal_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.abs(pre_warp_albedo - mask_albedo),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_sub_albedo_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)
                # save_image(
                #     torch.abs(pre_warp_normal - mask_normal),
                #     fp=os.path.join(self.path_save_plots,
                #                     'prediction_sub_normal_{}_{}.exr'.format(epoch,index_sequence)),
                #     format='exr',
                #     nrow=self.validation_dataloader.dataset.number_of_frames)

        print(f"Epoch {epoch}, PSNR: {psnr_sum / num:.4f}, SSIM: {ssim_sum / num:.4f}")
        print(f"Epoch {epoch}, CUR_CROPPED_PSNR: {cropped_psnr_sum / num:.4f}, CUR_CROPPED_SSIM: {cropped_ssim_sum / num:.4f}")
        with open(os.path.join(self.path_save_models, "valid_data.txt"), "a") as f:
            f.write(f"Epoch {epoch}, PSNR: {psnr_sum / num:.4f}, SSIM: {ssim_sum / num:.4f}\n")
            f.write(f"Epoch {epoch}, CUR_CROPPED_PSNR: {cropped_psnr_sum / num:.4f}, CUR_CROPPED_SSIM: {cropped_ssim_sum / num:.4f}\n")

    @torch.no_grad()
    def test(self, epoch = 0, plot_after_n_iterations = 1,
                 validation_metrics: Tuple[Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]]
                 = (nn.L1Loss(reduction='mean'), nn.MSELoss(reduction='mean'), misc.psnr, misc.ssim)) -> None:
        # Generator model to device
        self.setup_seed(9910)
        self.generator_network.to(self.device)
        self.generator_network.eval()
        # Main loop
        psnr_sum = 0
        ssim_sum = 0
        cropped_psnr_sum = 0
        cropped_ssim_sum = 0
        num = 0
        crop_size = 300
        pad_size = int(crop_size / 2)
        pre_warp_color = None
        grid = None
        grid_norm = None
        sample_time = None
        warped_id = None
        num_10000 = (torch.full((1, 1, self.height, self.width), -10000.0).to('cuda'))
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
        psnr = PeakSignalNoiseRatio().to('cuda')
        frame_index = 0

        # self.ssim_sub = []
        # with open('./ssim_tmp/ssim_0.txt', "r") as file:
        #     lines = file.readlines()
        #     for line in lines:
        #         self.ssim_sub.append(float(line))

        for index_sequence, batch in enumerate(self.test_dataloader):  # 遍历每个序列
            projected_color, cur_color, projected_sample_time, cur_sample_time, label_image, gaze_point = batch
            input_image = input_image.to(self.device)
            # albedo = albedo.to(self.device)
            # normal = normal.to(self.device)
            # grid = grid.to(self.device)
            sample_time_input = sample_time_input.to(self.device)
            label_image = label_image.to(self.device)
            # label_radiance = label_radiance.to(self.device)
            label_albedo = label_albedo.to(self.device)
            label_normal = label_normal.to(self.device)

            # b, c, h, w = label_image.shape
            # x = torch.linspace(-1, 1, steps=w + 1)[:-1] + 1 / w
            # y = torch.linspace(-1, 1, steps=h + 1)[:-1] + 1 / h
            # grid_y, grid_x = torch.meshgrid(y, x)
            # grid_norm = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).to(self.device)
                                        
            # grid_sub = grid - grid_norm
            # grid_sub[:, 0:1, :, :] *= self.width // 2
            # grid_sub[:, 1:2, :, :] *= self.height // 2

            # Make prediction
            prediction_image = self.generator_network(label_albedo, label_normal, input_image, sample_time_input)  # b c h w=1,18,h,w  1,6,h,w#generator要改channel
            # cur_gaze_coord = gaze_labels[0, 0:2]
            # cur_gaze_coord = [max(0, min(int(cur_gaze_coord[0]), 1919)), max(0, min(int(cur_gaze_coord[1]), 1079))]
            # pad_prediction = F.pad(prediction, (pad_size, pad_size, pad_size, pad_size))
            # pad_label_img = F.pad(label_img, (pad_size, pad_size, pad_size, pad_size))
            # cropped_prediction = pad_prediction[:, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
            # cropped_label_img = pad_label_img[:, :, cur_gaze_coord[1]:cur_gaze_coord[1]+crop_size, cur_gaze_coord[0]:cur_gaze_coord[0]+crop_size]
            # cropped_psnr_sum += psnr(cropped_prediction, cropped_label_img)
            # # print(psnr(cropped_prediction, cropped_label_img))
            # cropped_ssim_sum += ssim(cropped_prediction, cropped_label_img)
            # print(ssim(cropped_prediction, cropped_label_img))
            # print(f"Index {index_sequence}, CUR_CROPPED_PSNR: {psnr(cropped_prediction, cropped_label_img):.4f}, CUR_CROPPED_SSIM: {ssim(cropped_prediction, cropped_label_img):.4f}")
            # print(f"{index_sequence} {psnr(cropped_prediction, cropped_label_img):.4f} {ssim(cropped_prediction, cropped_label_img):.4f}")
            # with open(os.path.join(self.path_save_models, "valid_ssim.txt"), "a") as f:
            #     f.write(f"{ssim(cropped_prediction, cropped_label_img):.4f}\n")

            # print(f"{ssim(cropped_prediction, cropped_label_img):.4f}")

            prediction_gaze_position = torch.clamp(prediction_image[0] * 255, min=0, max=255).to(torch.uint8).cpu()
            cur_point = gaze_point[0]

            for thickness in range(0, 3):
                boxes = torch.tensor([[cur_point[0]-pad_size-thickness, cur_point[1]-pad_size-thickness, cur_point[0]+pad_size+thickness, cur_point[1]+pad_size+thickness]])  # 形状为(N, 4)，这里N=1
                prediction_gaze_position = draw_bounding_boxes(prediction_gaze_position, boxes, colors=[(255, 0, 0)])
                boxes = torch.tensor([[cur_point[0]-pad_size+thickness, cur_point[1]-pad_size+thickness, cur_point[0]+pad_size-thickness, cur_point[1]+pad_size-thickness]])  # 形状为(N, 4)，这里N=1
                prediction_gaze_position = draw_bounding_boxes(prediction_gaze_position, boxes, colors=[(255, 0, 0)])
            torch.cuda.synchronize()

            if index_sequence % plot_after_n_iterations == 0:
                save_image(
                    prediction_gaze_position.to(torch.float32) / 255,
                    fp=os.path.join(self.path_save_plots,
                                    'prediction_{}_{}.exr'.format(epoch,index_sequence)),
                    format='exr',
                    nrow=1)

    def inference(self, new_video, albedo, normal, label_fmv, label_img, high_objectId):
        """
        Inference method generates the reconstructed image to the corresponding input and saves the input, label and
        output as an image
        :param sequences: (List[torch.Tensor]) List of video sequences with shape
        [1 (batch size), 3 * 6 (rgb * frames), 192, 256]
        :param apply_fovea_filter: (bool) If true the fovea filter is applied to the input sequence
        """

