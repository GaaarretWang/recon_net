import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
from datetime import datetime,timedelta
#import time
#from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
#import matplotlib.pyplot as plt
#import time
import numpy as np
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def millis(start_time):
   dt = datetime.now() - start_time
   ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
   return ms
class Decoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Decoder, self).__init__()

    self.conv_relu = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

  def initialize_weights(self):
      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.xavier_normal_(m.weight.data)
              if m.bias is not None:
                  m.bias.data.zero_()
          elif isinstance(m, nn.BatchNorm2d):
              m.weight.data.fill_(1)
              m.bias.data.zero_()
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight.data, 0, 1)
              m.bias.data.zero_()


class MyNetNew(nn.Module):

    def __init__(self):
        super(MyNetNew, self).__init__()
        #self.base_model = torchvision.models.resnet18(True)
        #self.base_layers = list(self.base_model.children())
        self.TrianNumber = 1420
        self.Width = 1280
        self.Height = 720
        self.layer1 = nn.Sequential(
            #nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),#directpredict
            #nn.Conv2d(15, 48, kernel_size=3, stride=1, padding=1),#directGbuffer
            nn.Conv2d(13, 48, kernel_size=3, stride=1, padding=1),  # 加上warpPrev,去掉motionVector
            nn.ReLU(inplace=True),#(1,48,720，1280)
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),#(1,48,720，1280)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            #(1,48,360，640)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),#(1,48,360，640)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            #(1,48,180，320)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),#(1,48,180，320)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            #(1,48,90，160)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),#(1,48,90，160)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            #(1,48,45，80)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),# (1,48,45，80)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            # (1,48,22.5，40)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            #(1,48,22.5，40)
            )
        self.layer7 =nn.Upsample((45,80), mode='bilinear', align_corners=True)
        self.decode4 = Decoder(48 + 48, 96)

        self.decode4conv = nn.Sequential(#输入96，输出96
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer8 =nn.Upsample((90, 160), mode='bilinear', align_corners=True)

        self.decode3 = Decoder(96 + 48, 96)#输入96，当前输出96+layer3输出48，输出96
        self.decode3conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer9 =nn.Upsample((180, 320), mode='bilinear', align_corners=True)

        self.decode2 = Decoder(96 + 48, 96)
        self.decode2conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layer10 = nn.Upsample((360, 640), mode='bilinear', align_corners=True)

        self.decode1 = Decoder(96 + 48, 96)
        self.decode1conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        #self.decode1 = Decoder(96 + 48, 49*3)
        #self.decode1conv = nn.Sequential(
        #    nn.Conv2d(49*3, 49*3, kernel_size=3, stride=1, padding=1),
        #    nn.ReLU(inplace=True)
        #)
        self.layer11 = nn.Upsample((720, 1280), mode='bilinear', align_corners=True)

        #self.decode0 = Decoder(96 + 3, 64)#directoredict
        #self.decode0 = Decoder(96 + 15, 64)#directGbuffer和kenelpredict
        self.decode0 = Decoder(96 + 13, 64)  # 加上warpPrev

        #directpredict和directGbuffer
        #加上warpPrev
        #self.decode0conv = nn.Sequential(
        #    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        #    nn.ReLU(inplace=True)
        #)

        #kernelpredict
        self.decode0conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )#(1,25*2*3,1017,1920)这里的95是每个像素的5*5的核预测


        self.kernel1 = nn.Conv2d(64, 25*3*2+3, kernel_size=3, stride=1, padding=1)
        #self.kernel1 = nn.Conv2d(64, 25*2+1, kernel_size=3, stride=1, padding=1)# revise

        self.kernel2=nn.Conv2d(96, 25*3+3, kernel_size=3, stride=1, padding=1)
        #self.kernel2 = nn.Conv2d(96, 25+1, kernel_size=3, stride=1, padding=1)# revise

        self.kernel3=nn.Conv2d(96, 16*3+3, kernel_size=3, stride=1, padding=1)
        #self.kernel3 = nn.Conv2d(96, 25+1, kernel_size=3, stride=1, padding=1)# revise

        self.weight = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        )

        #self.direct = nn.Sequential(
        #    nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        #    nn.ReLU(inplace=True),
            # nn.InstanceNorm2d(1, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        #)
        self.kernel_pred = KernelConv(kernel_size = 5)
        self.kernel_pred2 = KernelConv(kernel_size = 5)
        self.kernel_pred3 = KernelConv(kernel_size = 4)
        #self.kernel_pred3 = KernelConv(kernel_size=5)#revise
        self.kernel_predp = KernelConv(kernel_size = 5)
        self.downscale2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.downscale3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.upscale2 = nn.Upsample((720, 1280), mode='bilinear', align_corners=True)
        self.upscale3 = nn.Upsample((360, 640), mode='bilinear', align_corners=True)

        self.upscale_nearest_23 = nn.Upsample((360, 640), mode='bilinear', align_corners=True)
        #self.downscale_23 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downscale_23 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
        self.upscale_nearest_12 = nn.Upsample((720, 1280), mode='bilinear', align_corners=True)
        #self.downscale_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downscale_12 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)

        self.allOne = torch.ones(1,3,720,1280).to(device)
        #self.allOne2 = torch.ones(1, 3, 360, 640).to(device)
        #self.allOne3 = torch.ones(1, 3, 180, 320).to(device)


    #def forward(self,input,noise,warp):#3,I,I [1, 3, 1017,1920] 假如只有一张I*I大小的图片的RGB通道
    def forward(self, input, noise,warp):  # 3,I,I [1, 3, 1017,1920] 假如只有一张I*I大小的图片的RGB通道
        start_time = datetime.now()
        e1 = self.layer1(input)  # 48,I/2,I/2  [1, 48, 508,960] 2*（卷积+Relu）+下采样 经过第一层后特征图大小
        e2 = self.layer2(e1)  # 48,I/4,I/4 [1, 48, 254,480] 卷积+Relu+下采样
        e3 = self.layer3(e2)  # 48,I/8,I/8 [1, 48, 127, 240] 卷积+Relu+下采样
        e4 = self.layer4(e3)  # 48,I/16,I/16 [1, 48, 63, 120] 卷积+Relu+下采样
        e5 = self.layer5(e4)  # 48,I/32,I/32 [1, 48, 31, 60] 卷积+Relu+下采样
        f = self.layer6(e5)  # 48,I/32,I/32 [1, 48, 31, 60] 卷积+Relu

        d4a = self.layer7(f) #[1, 48, 63, 120] 上采样
        d4b = self.decode4(d4a, e4)  # 48+48=96,I/16,I/16  unet结构的卷积+Relu
        d4c = self.decode4conv(d4b) # 96,I/16,I/16  卷积+Relu

        d3a = self.layer8(d4c) #[1, 96, 127, 240]上采样
        d3b = self.decode3(d3a, e3)  # 96,I/8,I/8 [1, 96, 240, 135]
        d3c = self.decode3conv(d3b)  # 96,I/8,I/8 [1, 96, 240, 135]

        d2a = self.layer9(d3c)# [1, 96, 254,480]上采样
        d2b = self.decode2(d2a, e2)  # 96,I/4,I/4
        d2c = self.decode2conv(d2b)  #96,I/4,I/4

        d1a = self.layer10(d2c)#[1, 96, 508, 960]上采样
        d1b = self.decode1(d1a, e1)  # 96,I/2,I/2
        d1c = self.decode1conv(d1b)  # 96,I/2,I/2
        #其实在这里d1c的输出就要是49*3了，之后在这里改改
        d0a = self.layer11(d1c)#上采样
        d0b = self.decode0(d0a, input)  # 64,I,I
        d0c = self.decode0conv(d0b)  # 64,I,I

        kernel1_weight = self.kernel1(d0c)

        #revise
        #kernel1_weight[:, :25, :, :] = F.softmax(kernel1_weight[:, :25, :, :],dim=1);
        #kernel1_weight[:, 25:50, :, :] = F.softmax(kernel1_weight[:, 25:50, :, :], dim=1);
        #kernel1_weight[:, 50:75, :, :] = F.softmax(kernel1_weight[:, 50:75, :, :], dim=1);

        kernel_noise = kernel1_weight[:,:75,:,:]
        #kernel_noise = kernel1_weight[:, :25, :, :]#revise

        #revise
        #kernel1_weight[:, 75:100, :, :] = F.softmax(kernel1_weight[:, 75:100, :, :],dim=1);
        #kernel1_weight[:, 100:125, :, :] = F.softmax(kernel1_weight[:, 100:125, :, :], dim=1);
        #kernel1_weight[:, 125:150, :, :] = F.softmax(kernel1_weight[:, 125:150, :, :], dim=1);

        kernel_warp = kernel1_weight[:,75:150,:,:]
        #kernel_warp = kernel1_weight[:, 25:50, :, :]#revise
        weight_warp = kernel1_weight[:,150:153,:,:]
        #weight_warp = kernel1_weight[:, 50:51, :, :]#revise

        kernel2_weight = self.kernel2(d1c)

        #revise
        #kernel2_weight[:, :25, :, :] = F.softmax(kernel2_weight[:, :25, :, :],dim=1);
        #kernel2_weight[:, 25:50, :, :] = F.softmax(kernel2_weight[:, 25:50, :, :], dim=1);
        #kernel2_weight[:, 50:75, :, :] = F.softmax(kernel2_weight[:, 50:75, :, :], dim=1);

        kernel_noise1_2 = kernel2_weight[:,:75,:,:]
        #kernel_noise1_2 = kernel2_weight[:, :25, :, :]#revise
        weight_noise1_2 = kernel2_weight[:,75:78,:,:]
        #weight_noise1_2 = kernel2_weight[:, 25:26, :, :]#revise

        kernel3_weight = self.kernel3(d2c)

        #revise
        #kernel3_weight[:, :16, :, :] = F.softmax(kernel3_weight[:, :16, :, :],dim=1);
        #kernel3_weight[:, 16:32, :, :] = F.softmax(kernel3_weight[:, 16:32, :, :], dim=1);
        #kernel3_weight[:, 32:48, :, :] = F.softmax(kernel3_weight[:, 32:48, :, :], dim=1);

        kernel3_noise1_4 = kernel3_weight[:,:48,:,:]
        #kernel3_noise1_4 = kernel3_weight[:, :25, :, :]#revise
        weight_noise1_4 = kernel3_weight[:,48:51,:,:]
        #weight_noise1_4 = kernel3_weight[:, 25:26, :, :]#revise

        # revise 添加softmax
        #kernel_noise=F.softmax(kernel_noise,dim=1);
        #kernel_warp=F.softmax(kernel_warp,dim=1);
        #kernel_noise1_2=F.softmax(kernel_noise1_2,dim=1);
        #kernel3_noise1_4=F.softmax(kernel3_noise1_4,dim=1);


        #direct = self.direct(d0b)#直接预测结果
        #weight = self.weight(d0c)#[1,6,720,1280]
        #weight_spatial = weight[:,0:3,:,:]
        #weight_temporal = weight[:, 3:6, :, :]

        extract = millis(start_time)
        extract_time = datetime.now()

        pred_i, pred=self.kernel_pred(noise, kernel_noise)
        pred_ip, pred_p=self.kernel_predp(warp, kernel_warp)
        halfNoise = self.downscale2(noise).requires_grad_()#将noise最大池化为1/2
        quaterNoise = self.downscale3(noise).requires_grad_()#将noise最大池化为1/4

        pred_i2, pred2= self.kernel_pred2(halfNoise, kernel_noise1_2)

        pred_i3, pred3 = self.kernel_pred3(quaterNoise, kernel3_noise1_4)

        kernelpredict = millis(extract_time)
        kernelpredict_time = datetime.now()

        image_fine = pred2.requires_grad_()
        image_course = pred3.requires_grad_()
        Dif = self.downscale_23(image_fine).requires_grad_()
        arphaDif = Dif.mul(weight_noise1_4).requires_grad_()
        arphaic = image_course.mul(weight_noise1_4).requires_grad_()
        UarphaDif = self.upscale_nearest_23(arphaDif).requires_grad_()
        Uarphaic = self.upscale_nearest_23(arphaic).requires_grad_()
        halflerp = (image_fine-UarphaDif+Uarphaic).requires_grad_()

        image_fine_o = pred.requires_grad_()
        image_course_o = halflerp.requires_grad_()
        Dif_o = self.downscale_12(image_fine_o).requires_grad_()
        arphaDif_o = Dif_o.mul(weight_noise1_2).requires_grad_()
        arphaic_o = image_course_o.mul(weight_noise1_2).requires_grad_()
        UarphaDif_o = self.upscale_nearest_12(arphaDif_o).requires_grad_()
        Uarphaic_o = self.upscale_nearest_12(arphaic_o).requires_grad_()

        lerpResult_spatial = (image_fine_o - UarphaDif_o + Uarphaic_o).requires_grad_()

        current = (self.allOne - weight_warp).mul(lerpResult_spatial).requires_grad_()
        histroy = weight_warp.mul(pred_p).requires_grad_()

        lerpResult = torch.add(current,histroy).requires_grad_()

        lerp = millis(kernelpredict_time)

        #return current,histroy, lerpResult,extract,kernelpredict,lerp

        return current, histroy, lerpResult

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        """
        print('---------- 打印网络信息 -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        #print(self)
        print('[网络] 总共有 : %.3f M 个数的参数' % (num_params / 1e6))
        print(list(self.parameters())[0].device)
        print('-----------------------------------------------')

    def save_networks(self, epoch, netname, savedir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        save_filename = '%s_%s.pth' % (netname, epoch)
        #save_path = os.path.join("./savemodels/directpredict", save_filename)
        save_path = os.path.join(savedir, save_filename)


        if torch.cuda.is_available():
            torch.save(self.cpu().state_dict(), save_path)
            self.cuda(device)
        else:
            torch.save(self.cpu().state_dict(), save_path)

    def load_networks(self, epoch, netname, loaddir):
        load_filename = '%s_%s.pth' % (netname, epoch)
        load_path = os.path.join(loaddir, load_filename)

        if torch.cuda.is_available():
            torch.load_state_dict(self.cpu().state_dict(), load_path)
            self.cuda(device)
        else:
            torch.load_state_dict(self.cpu().state_dict(), load_path)
        print()

    def save_pictures(self, epoch, iter, picturetensor, picturename, savedir):
        save_filename = 'epoch%s_iter%s_%s.png' % (epoch, iter, picturename )
        save_path = os.path.join(savedir, save_filename)
        save_image(picturetensor, save_path)

class KernelConv(nn.Module):
    def __init__(self, kernel_size=7):
        super(KernelConv, self).__init__()
        self.kernel_size = kernel_size

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width (1,18*3*3,1017,1920)
        :return: core_out, a dict
        """

        #revise
        #1,25,720,1280
        #core=core.unsqueeze(0);
        #1,1,25,720,1280
        #core = core.unsqueeze(3);
        #1,1,25,1,720,1280
        #core = torch.cat((core, core, core), dim=3)
        #1,1,25,3,720,1280

        core = core.view(batch_size, N, -1, color, height, width)#1 2 25 3 192 256
        #-1 表示 PyTorch 自动计算缺失的维度大小

        core=F.softmax(core,dim=2)#对5*5的卷积核的权重做softmax，将其归一化在[0,1]
        return core

    #def forward(self, frames, warp, core):
    def forward(self, frames,core):#frame:1 3*2 192 256,core:1 25*3*2 192 256
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]  1,1,3,720,1280
        :param core: [batch_size, N, dict(kernel), 3, height, width]#1,1,25,3,720,1280
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 3
            N=1#N=2
            frames = frames.view(batch_size, N, color, height, width)#  1 2 3 192 256


        core = self._convert_dict(core, batch_size, N, color, height, width)#（1,2,25,3,192,256）


        img_stack = []
        pred_img = []

        kernel = self.kernel_size
        if not img_stack:
            frame_pad = F.pad(frames, [kernel // 2, kernel // 2, kernel // 2, kernel // 2])#在输入图像周围加一圈宽2的像素
            #frame_pad:1,2,3,196,260
            for i in range(kernel):
                for j in range(kernel):

                    img_stack.append(frame_pad[..., i:i + height, j:j + width])
                    #img_stack.append(warp_pad[..., i:i + height, j:j + width])
            #with torch.no_grad():
            img_stack = torch.stack(img_stack, dim=2)
            #img_stack 1,1,49,3,1017,1920
            #print('img_stack:', img_stack.size())

        #core (1,1,25,1,720,1280)
        #img_stack (1,1,25,3,720,1280)
        pred_img.append(torch.sum(core.mul(img_stack), dim=2, keepdim=False))

        pred_img = torch.stack(pred_img, dim=0)
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
        return pred_img_i, pred_img
       #pred_img_i应该是[1,1,3,1017,1920]， pred_img应该是[1，3，1017，1920]

class LossFunc(nn.Module):
    """
    loss function of KPN
    """
    def __init__(self, coeff_basic=1.0, coeff_anneal=1.0, gradient_L1=True, alpha=0.9998, beta=100):
        super(LossFunc, self).__init__()
        self.coeff_basic = coeff_basic
        self.coeff_anneal = coeff_anneal
        #self.loss_basic = LossBasic(gradient_L1)
        #self.loss_anneal = LossAnneal(alpha, beta)
        self.loss_asymmetric = LossAsym()

        self.loss_temporal = nn.L1Loss().to(device)
        self.loss_spatial = nn.L1Loss().to(device)
        self.loss_maskL1 = LossMask()

    #def forward(self, denoise, reference, prevRef, prevDenoise,noise):
    def forward(self,denoise,reference,prevRef,prevDenoise,noise,traMV,occMV,preTraMV1,preOccMV1,preTraMV2,preOccMV2):
        """
        forward function of loss_func
        :param frames: frame_1 ~ frame_N, shape: [batch, N, 3, height, width]
        :param core: a dict coverted by ......
        :param ground_truth: shape [batch, 3, height, width]
        :param global_step: int
        :return: loss
        """
        return self.loss_spatial(denoise,reference),self.loss_temporal(denoise-prevDenoise,reference-prevRef),self.loss_maskL1(denoise,reference,traMV,occMV,preTraMV1,preOccMV1,preTraMV2,preOccMV2)
        #return self.loss_spatial(denoise,reference),self.loss_temporal(denoise-prevDenoise,reference-prevRef)

class LossBasic(nn.Module):
    """
    Basic loss function.
    """
    #def __init__(self, gradient_L1=True):
    def __init__(self):
        super(LossBasic, self).__init__()
        self.l1_loss = nn.L1Loss()
        #self.l2_loss = nn.MSELoss()
        #self.gradient = TensorGradient(gradient_L1)

    def forward(self, pred, ground_truth):
        #return self.l2_loss(pred, ground_truth) + \
        #       self.l1_loss(self.gradient(pred), self.gradient(ground_truth))
        return self.l1_loss(pred,ground_truth)


class LossAsym(nn.Module):
    """
    anneal loss function
    """
    def __init__(self):
        super(LossAsym, self).__init__()
        #self.global_step = 0
        #self.loss_func = LossBasic(gradient_L1=True)
        #self.loss_func = LossBasic()
        #self.alpha = alpha
        #self.beta = beta

    def forward(self, denoise, reference, noise, traMV):

        deltaRN = reference - noise
        deltaDR = denoise - reference

        #逐像素乘法star*星乘
        DRstarRN = deltaRN*deltaDR
        #把负值设为0 #把非零值设为1
        DRstarRN = torch.where(DRstarRN<= -0., 0., 1.)
        #traMV= 50*(traMV-0.5)#把motionVector的数值恢复成原样
        #如果motionVector中接近0的值，也就是静止不动的地方，设置为1
        #其他不为0的值，也就是正在运动的地方，设置为0
        traMV = torch.where(abs(traMV)<1.0, 1., 0.)
        #全1矩阵
        allOne = torch.ones(1, 3,720,1280).to(device='cuda')

        DRRNstarMV = allOne+10*DRstarRN*traMV
        #损失矩阵（按照像素位置的reference和denoise之差）
        lossL1matrix = abs(reference-denoise)*DRRNstarMV

        #对损失矩阵求均值，是否相当于求L1损失？
        loss = lossL1matrix.mean()
        return loss

class LossMask(nn.Module):
    """
    anneal loss function
    """
    def __init__(self):
        super(LossMask, self).__init__()

    def forward(self,denoise,reference,traMV,currentOccMV,preTraMV1,preOccMV1,preTraMV2,preOccMV2):

        #traMV = torch.where(abs(traMV) > 0., 1., 0.)
        #currentOccMV = torch.where(abs(currentOccMV) > 0., 1., 0.)
        #preTraMV1 = torch.where(abs(preTraMV1) > 0., 1., 0.)
        #preOccMV1 = torch.where(abs(preOccMV1) > 0., 1., 0.)
        #preTraMV2 = torch.where(abs(preTraMV2) > 0., 1., 0.)
        #preOccMV2 = torch.where(abs(preOccMV2) > 0., 1., 0.)

        mask_0 = currentOccMV - traMV
        mask_1 = preOccMV1 - preTraMV1
        mask_2 = preOccMV2 - preTraMV2

        mask_0 = torch.where(abs(mask_0) > 0.5, 1., 0.)
        mask_1 = torch.where(abs(mask_1) > 0.5, 1., 0.)
        mask_2 = torch.where(abs(mask_2) > 0.5, 1., 0.)


        mask00 = mask_0[:,0:1,:,:]#x
        mask01 = mask_0[:,1:,:,:]#y
        mask0 = mask00+mask01
        mask0 = torch.cat((mask0,mask0,mask0),dim=1)
        mask0= torch.where(abs(mask0) > 0., 1., 0.)#无论x还是y，只要有移动就mask

        mask10 = mask_1[:,0:1,:,:]
        mask11 = mask_1[:,1:,:,:]
        mask1 = mask10+mask11
        mask1 = torch.cat((mask1,mask1,mask1),dim=1)
        mask1 = torch.where(abs(mask1) > 0., 0.8, 0.)

        mask20 = mask_2[:,0:1,:,:]
        mask21 = mask_2[:,1:,:,:]
        mask2 = mask20+mask21
        mask2 = torch.cat((mask2,mask2,mask2),dim=1)
        mask2 = torch.where(abs(mask2) > 0., 0.6, 0.)

        mask = mask0+mask1+mask2


        #lossL1matrix0 = abs(reference - denoise) * mask0
        #lossL1matrix1 = abs(reference - denoise) * mask1
        #lossL1matrix2 = abs(reference - denoise) * mask2

        lossL1matrix = abs(reference - denoise) * mask
        lossL1matrix[0,0,0,0]=0.0001#为了不让所有值为0导致loss索引nan

        #loss0 = lossL1matrix0.mean()
        #loss1 = lossL1matrix1.mean()
        #loss2 = lossL1matrix2.mean()

        loss = lossL1matrix[lossL1matrix>0].mean()

        #loss = lossL1matrix[lossL1matrix > 0.1].mean()

        if(loss <0.002):
            loss = 0.0;


        #loss = lossL1matrix.mean()*50.0

        #loss = lossL1matrix.sum()
        #print(loss)
        #print(loss)
        return loss



class TensorGradient(nn.Module):
    """
    the gradient of tensor
    """
    def __init__(self, L1=True):
        super(TensorGradient, self).__init__()
        self.L1 = L1

    def forward(self, img):
        w, h = img.size(-2), img.size(-1)
        l = F.pad(img, [1, 0, 0, 0])
        r = F.pad(img, [0, 1, 0, 0])
        u = F.pad(img, [0, 0, 1, 0])
        d = F.pad(img, [0, 0, 0, 1])
        if self.L1:
            return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])
        else:
            return torch.sqrt(
                torch.pow((l - r)[..., 0:w, 0:h], 2) + torch.pow((u - d)[..., 0:w, 0:h], 2)
            )

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



#log_dir = "./net/Unet"
#writer = SummaryWriter(comment='test_Unet', filename_suffix="_test_Unet")
#path_img = "./mysequence/animePica/reference/result-0001.png"


#img_transforms = transforms.Compose([
    #transforms.ToTensor(),
#])
#img_pil = Image.open(path_img).convert('RGB')
#if img_transforms is not None:
    #img_tensor = img_transforms(img_pil)
#img_tensor.unsqueeze_(0)    # chw --> bchw

#input_img = torch.randn(1, 3, 1920, 1080)
#fake_img = fake_img


#ntkpnet = NTKPnet()
#output_img = ntkpnet(img_tensor)
#output_img.transpose_(0, 1)
#fmap_1_grid = vutils.make_grid(output_img, normalize=True, scale_each=True, nrow=8)
#writer.add_image('feature map', fmap_1_grid)

#plt.subplot(121).hist(input_img, label="union")
#plt.subplot(122).hist(output_img, label="normal")
#plt.legend()
#plt.show()

#writer.add_graph(ntkpnet, img_tensor)
#input_img = input_img.squeeze()
#writer.add_image("input_img", input_img, dataformats="CWH")
#output_img = output_img.squeeze()


#print(input_img)
#print(output_img)
#img_tensor=img_tensor.squeeze(0)
#writer.add_image("img_tensor", img_tensor, 2)
#writer.close()