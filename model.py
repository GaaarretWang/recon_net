from typing import Tuple, Union, Type, List

from utils import upsample_zero_2d,backwarp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from modules.modulated_deform_conv import ModulatedDeformConvPack
from torchvision.ops.deform_conv import DeformConv2d
from base import BaseModel
import time
import torch.nn.init as init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_0_1_batch(input: torch.tensor) -> torch.tensor:
    '''
    Normalize a given tensor to a range of [0, 1]
    Source: https://github.com/ChristophReich1996/Semantic_Pyramid_for_Image_Generation/blob/master/misc.py

    :param input: (Torch tensor) Input tensor
    :return: (Torch tensor) Normalized output tensor
    '''
    input_flatten = input.reshape(input.shape[0], -1)
    return ((input - torch.min(input_flatten, dim=1)[0][:, None, None, None]) / (
            torch.max(input_flatten, dim=1)[0][:, None, None, None] -
            torch.min(input_flatten, dim=1)[0][:, None, None, None]))

class dcn(nn.Module):
    def __init__(self, in_channels, out_channels):  
        super(dcn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) #原卷积
        self.conv_offset = nn.Conv2d(in_channels, 18, kernel_size=3, stride=1, padding=1)
        self.conv_mask = nn.Conv2d(in_channels, 9, kernel_size=3, stride=1, padding=1)
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                             mask=mask, padding=(1, 1))
        return out

class RecurrentUNet1(nn.Module):
    """
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    """

    def __init__(self) -> None:
        """
        Constructor method
        :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
        :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
        :param channels_super_resolution_blocks: (Tuple[Tuple[int, int]]) In and out channels in each s.r. block
        """
        # Call super constructor
        super(RecurrentUNet1, self).__init__()  # 调用了父类nn.Module的构造函数
        self.pre_feature_extraction_model = NSRRFeatureExtractionModel()
        # self.feature_reweighting_model = NSRRFeatureReweightingModel()

        
        self.feature_extraction_conv = Feature_Extraction_RecurrentUNet()
        self.pre_warp_conv = Pre_Warp_RecurrentUNet()
        self.pre_warp_conv_cur = Pre_Warp_RecurrentUNet_Cur()
        self.pre_recon = Pre_Recon()

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.kernel_recon = KernelConv(kernel_size=3)
        self.kernel_pred = KernelConv(kernel_size=3)
        self.kernel_pred1 = KernelConv1(kernel_size=[3])
        self.kernel_recon1 = KernelConv1(kernel_size=[3])

        # self.allOne_low = torch.ones(1, 1, heihgt, 480).to(device)

    def forward(self, projected_color, cur_color, projected_sample_time, cur_sample_time, pre_prediction):  # 1 24 192 256
        """
        Forward pass
        :param input: (torch.Tensor) Input frame
        :return: (torch.Tensor) Super resolution output frame
        input:1 12 192 256
        noise:1 3 192 256
        warp:1 3 192 256:down_warp
        HR_Warp:1 3 768 1024
        warp_depth:1 1 192 256:mask_depth
        """
        # down_warp = self.pixel_unshuffle(samewarp)  # 1 3 270 480

        # samewarp = (self.allOne - mv_mask) * recon_results + mv_mask * samewarp
        # samewarp = mv_mask * samewarp
        # feature_down7, feature_layer0, feature_decode7_output = self.feature_extraction_conv(torch.cat((down_warp, mask_albedo, mask_depth), dim=1))
        # pre_warp_color_unshuffle = self.pixel_unshuffle(pre_warp_color)  # 1 3 270 480
        # sample_time_input_unshuffle = self.pixel_unshuffle(sample_time_input)  # 1 3 270 480
        # pre_warp_color_unshuffle_conv = self.pre_recon(torch.cat((pre_warp_color_unshuffle, sample_time_input_unshuffle, mask_albedo, mask_depth), dim=1))
        # pre_warp_color_shuffle = self.pixel_shuffle(pre_warp_color_unshuffle_conv)  # 1 3 270 480
        # features = self.pre_feature_extraction_model(torch.cat((sample_time_input), dim=1))
        # offset = self.gen_offset(torch.cat((sample_time_input, grid_sub, albedo, normal, label_albedo, label_normal), dim=1))
        # result = self.pre_warp_conv(torch.cat((projected_color, cur_color, projected_sample_time, cur_sample_time), dim=1))
        result = self.pre_warp_conv_cur(torch.cat((cur_color, cur_sample_time), dim=1))
        result = self.pre_warp_conv(torch.cat((projected_color, result, projected_sample_time, pre_prediction), dim=1))

        # recon5 = torch.ones_like(decode5_color)
        # recon6 = torch.ones_like(decode6_color)
        # kernel_weight7 = conv7[:, 2:11, :, :]
        # # pred_color = conv7[:, 9:12, :, :]
        # pred_color = down7[:, 0:3, :, :]
        # recon7 = torch.ones_like(pred_color)


        # recon5 = F.interpolate(recon5, scale_factor=4, mode='bilinear', align_corners=False)
        # # recon6 = F.interpolate(decode6_color, scale_factor=1/2, mode='bilinear', align_corners=False)
        # recon6 = F.interpolate(recon6, scale_factor=2, mode='bilinear', align_corners=False)

        # # pred_color1 = kernel_weight[:, 0:3, :, :]  # 1 3 1080 1920
        # # pred_color2 = kernel_weight[:, 3:6, :, :]  # 1 3 1080 1920
        # weight_recon0 = conv7[:, 0:1, :, :]  # 1 3 1080 1920
        # # weight_recon0 = (torch.sin(weight_recon0) + self.allOne) / 2
        # result0 = weight_recon0 * recon5 + (self.allOne - weight_recon0) * recon6
        # weight_recon1 = conv7[:, 1:2, :, :]  # 1 3 1080 1920
        # # weight_recon1 = (torch.sin(weight_recon1) + self.allOne) / 2
        # result = weight_recon1 * result0 + (self.allOne - weight_recon1) * recon7

        return result

class Pre_Warp_RecurrentUNet(nn.Module):  
    """  
    This class implements a recurrent U-Net to perform super-resolution based on the DeepFovea architecture.  
    """  

    def __init__(self,  
                 input_channels=10,  
                 label_channels=3,  
                 base_channels=16,  
                 encoder_channels=(12, 18, 36, 48, 60),  
                 unshuffle_channels=(12, 18, 36, 48),  
                 decoder_channels=(48, 36, 18, 12),  
                 kernel_size=3):  
        """  
        Constructor method.  
        """  
        super(Pre_Warp_RecurrentUNet, self).__init__()  

        # Initial convolutional layers  
        self.initial_conv = nn.Sequential(  
            nn.Conv2d(input_channels, encoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  

        self.pool1 = nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=2, stride=2)  

        # Encoder block 2  
        self.encoder2 = nn.Sequential(  
            nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[1], encoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  
        self.pool2 = nn.Conv2d(encoder_channels[1], encoder_channels[1], kernel_size=2, stride=2)  

        # Encoder block 3  
        self.encoder3 = nn.Sequential(  
            nn.Conv2d(encoder_channels[1], encoder_channels[2], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[2], encoder_channels[2], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  
        self.pool3 = nn.Conv2d(encoder_channels[2], encoder_channels[2], kernel_size=2, stride=2)  

        # Encoder block 4  
        self.encoder4 = nn.Sequential(  
            nn.Conv2d(encoder_channels[2], encoder_channels[3], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[3], encoder_channels[3], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  
        self.pool4 = nn.Conv2d(encoder_channels[3], encoder_channels[3], kernel_size=2, stride=2)  

        # Bottleneck  
        self.bottleneck = nn.Sequential(  
            nn.Conv2d(encoder_channels[3], encoder_channels[4], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[4], encoder_channels[4], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  

        # Decoder blocks  
        self.upconv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.decoder4 = nn.Sequential(  
            nn.Conv2d(encoder_channels[4] + encoder_channels[3], decoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ELU(inplace=True),  
        )  

        self.upconv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.decoder3 = nn.Sequential(  
            nn.Conv2d(decoder_channels[0] + encoder_channels[2], decoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(decoder_channels[1], decoder_channels[1] + 9, kernel_size=kernel_size, padding=1),  
            nn.ELU(inplace=True),  
        )  

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.decoder2 = nn.Sequential(  
            nn.Conv2d(decoder_channels[1] + encoder_channels[1], decoder_channels[2], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(decoder_channels[2], decoder_channels[2], kernel_size=kernel_size, padding=1),  
            nn.ELU(inplace=True),  
        )  

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.decoder1 = nn.Sequential(  
            nn.Conv2d(decoder_channels[2] - 9 + 7, decoder_channels[3], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=kernel_size, padding=1),  
            nn.ELU(inplace=True),  
        )  

        # Kernel prediction layers (assuming KernelConv1 is defined elsewhere)  
        self.kernel_pred1 = KernelConv1(kernel_size=[3])  
        self.kernel_recon1 = KernelConv1(kernel_size=[3])  

    def forward(self, input_tensor):  
        """  
        Forward pass of the network.  
        :param input_tensor: Input tensor.  
        :param label_albedo: Albedo label tensor.  
        :return: Output tensor.  
        """  
        # Initial convolution  
        down1 = self.initial_conv(input_tensor)  
        down1_pooled = self.pool1(down1)  

        # Encoder blocks  
        down2 = self.encoder2(down1_pooled)  
        down2_pooled = self.pool2(down2)  

        down3 = self.encoder3(down2_pooled)  
        down3_pooled = self.pool3(down3)  

        down4 = self.encoder4(down3_pooled)  
        down4_pooled = self.pool4(down4)  

        # Bottleneck  
        bottleneck = self.bottleneck(down4_pooled)

        # Decoder blocks  
        up4 = self.upconv4(bottleneck)  
        decode4_output = self.decoder4(torch.cat((up4, down4), dim=1))  

        up3 = self.upconv3(decode4_output)  
        decode5_output = self.decoder3(torch.cat((up3, down3), dim=1))  
        decode5_kernel = decode5_output.clone()  
        batch_size, num_channels, height, width = decode5_kernel.size()  

        # Kernel predictions for decode5_output  
        core5, bias5 = self.kernel_pred1._convert_dict(decode5_output[:, -9:, :, :], batch_size, 1, 1, height, width)  
        for i in range(3):  
            decode5_kernel[:, i:i+1, :, :] = self.kernel_pred1(decode5_output[:, i:i+1, :, :], core5, bias5)  

        up2 = self.upconv2(decode5_kernel[:, :-9, :, :])  
        decode6_output = self.decoder2(torch.cat((up2, down2), dim=1))  
        decode6_kernel = decode6_output.clone()  
        batch_size, num_channels, height, width = decode6_kernel.size()  

        # Kernel predictions for decode6_output  
        core6, bias6 = self.kernel_pred1._convert_dict(decode6_output[:, -9:, :, :], batch_size, 1, 1, height, width)  
        for i in range(3):  
            decode6_kernel[:, i:i+1, :, :] = self.kernel_pred1(decode6_output[:, i:i+1, :, :], core6, bias6)  

        up1 = self.upconv1(decode6_kernel[:, :-9, :, :])  
        decode7_input = torch.cat((down1[:, :7, :, :], up1), dim=1)  
        decode7_output = self.decoder1(decode7_input)  
        decode7_kernel = decode7_output.clone()  
        batch_size, num_channels, height, width = decode7_kernel.size()  

        # Kernel predictions for decode7_output  
        core7, bias7 = self.kernel_pred1._convert_dict(decode7_output[:, -9:, :, :], batch_size, 1, 1, height, width)  
        for i in range(3):  
            decode7_kernel[:, i:i+1, :, :] = self.kernel_recon1(decode7_output[:, i:i+1, :, :], core7, bias7)  

        return decode7_kernel[:, 0:3, :, :]  
    
class Pre_Warp_RecurrentUNet_Cur(nn.Module):  
    """  
    This class implements a recurrent U-Net to perform super-resolution based on the DeepFovea architecture.  
    """  

    def __init__(self,  
                 input_channels=64,  
                 label_channels=3,  
                 base_channels=16,  
                 encoder_channels=(128, 256, 512),  
                 unshuffle_channels=(2, 18, 36, 48),  
                 decoder_channels=(256, 48),  
                 kernel_size=3):  
        """  
        Constructor method.  
        """  
        super(Pre_Warp_RecurrentUNet_Cur, self).__init__()  

        # Initial convolutional layers  
        self.initial_conv = nn.Sequential(  
            nn.Conv2d(input_channels, encoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  

        self.pool1 = nn.Conv2d(encoder_channels[0], encoder_channels[0], kernel_size=2, stride=2)  

        # Encoder block 2  
        self.encoder2 = nn.Sequential(  
            nn.Conv2d(input_channels + encoder_channels[0], encoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[1], encoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  
        self.pool2 = nn.Conv2d(encoder_channels[1], encoder_channels[1], kernel_size=2, stride=2)  

        # Bottleneck  
        self.bottleneck = nn.Sequential(  
            nn.Conv2d(input_channels + encoder_channels[1], encoder_channels[2], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(encoder_channels[2], encoder_channels[2], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
        )  

        self.upconv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.decoder2 = nn.Sequential(  
            nn.Conv2d(input_channels + encoder_channels[2] + encoder_channels[1], decoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=kernel_size, padding=1),  
            nn.ELU(inplace=True),  
        )  

        self.upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.decoder1 = nn.Sequential(  
            nn.Conv2d(input_channels + decoder_channels[0] + encoder_channels[0], decoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(decoder_channels[1], decoder_channels[1], kernel_size=kernel_size, padding=1),  
            nn.ELU(inplace=True),  
        )  

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_shuffle = nn.PixelShuffle(4)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  

    def forward(self, input_tensor):  
        """  
        Forward pass of the network.  
        :param input_tensor: Input tensor.  
        :param label_albedo: Albedo label tensor.  
        :return: Output tensor.  
        """  
        # Initial convolution  
        input_tensor = self.pixel_unshuffle(input_tensor)
        input_tensor_2 = self.maxpool(input_tensor)
        input_tensor_4 = self.maxpool(input_tensor_2)

        down1 = self.initial_conv(input_tensor)  
        down1_pooled = self.pool1(down1)  

        # Encoder blocks  
        down2 = self.encoder2(torch.cat((input_tensor_2, down1_pooled), dim=1))  
        down2_pooled = self.pool2(down2)  

        # Bottleneck  
        bottleneck = self.bottleneck(torch.cat((input_tensor_4, down2_pooled), dim=1))
        # bottleneck = torch.cat((bottleneck, down2_pooled), dim=1)


        up2 = self.upconv2(bottleneck)  
        decode6_output = self.decoder2(torch.cat((input_tensor_2, up2, down2), dim=1))  

        up1 = self.upconv1(decode6_output)  
        decode7_output = self.decoder1(torch.cat((input_tensor, down1, up1), dim=1))  

        decode7_output = self.pixel_shuffle(decode7_output)

        return decode7_output

class Pre_Recon(nn.Module):
    """
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    """

    def __init__(self) -> None:
        """
        Constructor method
        :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
        :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
        :param channels_super_resolution_blocks: (Tuple[Tuple[int, int]]) In and out channels in each s.r. block
        """
        # Call super constructor
        super(Pre_Recon, self).__init__()  # 调用了父类nn.Module的构造函数

        self.layer11 = nn.Sequential(
            nn.Conv2d(68, 40, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )

        self.downlayer7 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
            nn.Conv2d(40, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )

        self.layer0 = nn.Sequential(
            nn.Conv2d(32 + 40, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )

        self.decode7conv = nn.Sequential(
            nn.Conv2d(32 + 32 + 40, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )

    def forward(self, features):  # 1 24 192 256
        """
        Forward pass
        :param input: (torch.Tensor) Input frame
        :return: (torch.Tensor) Super resolution output frame
        input:1 12 192 256
        noise:1 3 192 256
        warp:1 3 192 256:down_warp
        HR_Warp:1 3 768 1024
        warp_depth:1 1 192 256:mask_depth
        """
        pre_merge = self.layer11(features)
        down7 = self.downlayer7(pre_merge)

        layer0 = self.layer0(torch.cat((down7, pre_merge), dim=1))

        up7 = self.decode7conv(torch.cat((layer0, down7, pre_merge), dim=1))

        return up7

class Feature_Extraction_RecurrentUNet(nn.Module):
    """
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    """

    def __init__(self) -> None:
        """
        Constructor method
        :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
        :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
        :param channels_super_resolution_blocks: (Tuple[Tuple[int, int]]) In and out channels in each s.r. block
        """
        # Call super constructor
        super(Feature_Extraction_RecurrentUNet, self).__init__()  # 调用了父类nn.Module的构造函数

        self.downlayer7 = nn.Sequential(
            nn.Conv2d(52, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
            nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )
        self.downlayer7_down = nn.Conv2d(12, 12, kernel_size=2, stride=2, padding=0)

        # self.downlayer6 = nn.Sequential(
        #     nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),  # (1,24,720，1280)
        #     nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),  # (1,24,720，1280)
        # )
        # self.downlayer6_down = nn.AvgPool2d(2)

        # self.downlayer5 = nn.Sequential(
        #     nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),  # (1,24,720，1280)
        #     nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),  # (1,24,720，1280)
        # )
        # self.downlayer5_down = nn.AvgPool2d(2)

        self.layer0 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
            nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )

        # self.uplayer5 = nn.Upsample(size=(135,240), mode='bilinear', align_corners=True)#(270,480)
        # self.decode5conv = nn.Sequential(
        #     nn.Conv2d(16 + 12, 12, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(12, 12, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),  # (1,24,720，1280)
        # )

        # self.uplayer6 = nn.Upsample(size=(270,480), mode='bilinear', align_corners=True)#(270,480)
        # self.decode6conv = nn.Sequential(
        #     nn.Conv2d(5 + 5, 5, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),  # (1,24,720，1280)
        # )

        self.uplayer7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#(270,480)
        self.decode7conv = nn.Sequential(
            nn.Conv2d(12 + 12, 18, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(18, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,24,720，1280)
        )

    def forward(self, features):  # 1 24 192 256
        """
        Forward pass
        :param input: (torch.Tensor) Input frame
        :return: (torch.Tensor) Super resolution output frame
        input:1 12 192 256
        noise:1 3 192 256
        warp:1 3 192 256:down_warp
        HR_Warp:1 3 768 1024
        warp_depth:1 1 192 256:mask_depth
        """
        down7 = self.downlayer7(features)
        down7_down = self.downlayer7_down(down7)

        layer0 = self.layer0(down7_down)

        up7 = self.uplayer7(layer0)
        decode7_output = self.decode7conv(torch.cat((up7, down7), dim=1))

        return down7, layer0, decode7_output


class RecurrentUNet(nn.Module):
    """
    This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
    """

    def __init__(self) -> None:
        """
        Constructor method
        :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
        :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
        :param channels_super_resolution_blocks: (Tuple[Tuple[int, int]]) In and out channels in each s.r. block
        """
        # Call super constructor
        super(RecurrentUNet, self).__init__()  # 调用了父类nn.Module的构造函数
        self.pre_feature_extraction_model = NSRRFeatureExtractionModel()
        self.feature_reweighting_model = NSRRFeatureReweightingModel()
        self.layer1 = nn.Sequential(
            # nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),#directpredict
            # nn.Conv2d(15, 48, kernel_size=3, stride=1, padding=1),#directGbuffer
            nn.Conv2d(12 + 12, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,48,720，1280)
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,48,720，1280)

            # (1,48,360，640)
        )
        self.layer1_down = nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)

        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,48,360，640)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            # (1,48,180，320)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,48,180，320)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            # (1,48,90，160)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,48,90，160)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            # (1,48,45，80)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # (1,48,45，80)
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0)
            # (1,48,22.5，40)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # (1,48,22.5，40)
        )

        self.layer7 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#(16,30)
        self.decode4 = Decoder(48 + 48, 96)
        self.decode4conv = nn.Sequential(  # 输入96，输出96
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer8 = nn.Upsample(size=(33,60), mode='bilinear', align_corners=True)#(33,60)
        self.decode3 = Decoder(96 + 48, 96)  # 输入96，当前输出96+layer3输出48，输出96
        self.decode3conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer9 = nn.Upsample(size=(67,120), mode='bilinear', align_corners=True)#(67,120)
        self.decode2 = Decoder(96 + 48, 96)
        self.decode2conv = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer10 = nn.Upsample(size=(135,240), mode='bilinear', align_corners=True)#(135,240)
        self.decode1 = Decoder(96 + 48, 32)
        self.decode1conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.layer11 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#(270,480)
        self.decode0 = Decoder(32 + 24, 48)  # 加上warpPrev
        self.decode0conv = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dconv1 = SuperResolutionBlock(in_channels=48 + 48, out_channels=32)
        self.dconv2 = SuperResolutionBlock(in_channels=32 + 48, out_channels=24)

        self.kernel_pred = KernelConv(kernel_size=3)
        self.kernel_predp = KernelConv(kernel_size=3)
        self.kernel_pred2 = KernelConv(kernel_size=3)
        self.kernel_pred3 = KernelConv(kernel_size=4)

        self.allOne = torch.ones(1, 3, 270*4, 480*4).to(device)



    def forward(self, input: torch.Tensor,
                noise: torch.Tensor,
                warp: torch.Tensor ,
                HR_Warp:torch.Tensor,
                warp_depth: torch.Tensor,
                mv_mask: torch.Tensor):  # 1 24 192 256
        """
        Forward pass
        :param input: (torch.Tensor) Input frame
        :return: (torch.Tensor) Super resolution output frame
        input:1 12 192 256
        noise:1 3 192 256
        warp:1 3 192 256:down_warp
        HR_Warp:1 3 768 1024
        warp_depth:1 1 192 256:mask_depth
        """

        # torch.cuda.synchronize()
        # time_en_s = time.perf_counter()

        pre_features = self.pre_feature_extraction_model.forward(warp, warp_depth)#1 12 192 256
        pre_features_reweighted = self.feature_reweighting_model.forward(
            input,  # 1 12 192 256
            pre_features  # 1 12 192 256
        )  # 1 12 192 256
        # print('pre_features_reweighted:', pre_features_reweighted.shape)


        final_input = torch.cat((input,pre_features_reweighted),dim=1)#1 24 192 256

        # warp_reweight = pre_features_reweighted[:,:3,:,:]#1 3 192 256
        # print('pre_features_reweighted:', pre_features_reweighted.shape)
        # print('final_input:', final_input.shape)
        # print('warp_reweight:', warp_reweight.shape)
        e1u = self.layer1(final_input)  # 1 48 270 480
        # print('e1u:', e1u.shape)
        e1 = self.layer1_down(e1u)  # 1 48 135 240
        # print('e1:' , e1.shape)
        e2 = self.layer2(e1)  # 1 48 67 120
        # print('e2:', e2.shape)
        e3 = self.layer3(e2)  # 1 48 33 60
        # print('e3:', e3.shape)
        e4 = self.layer4(e3)  # 1 48 16 30
        # print('e4:', e4.shape)
        e5 = self.layer5(e4)  # 1 48 8 15
        # print('e5:', e5.shape)
        f = self.layer6(e5)  # 1 48 8 15
        # print('f:', f.shape)

        d4a = self.layer7(f)  # 1 48 16 30
        d4b = self.decode4(d4a, e4)  # 48+48->96,16,30
        d4c = self.decode4conv(d4b)  # 1,96,16,30

        d3a = self.layer8(d4c)  # 1 96 33 60
        d3b = self.decode3(d3a, e3)  # 1,96+48->96,33,60
        d3c = self.decode3conv(d3b)  # 1,96,24,32

        d2a = self.layer9(d3c)  # 1 96 67 120
        d2b = self.decode2(d2a, e2)  # 1,96+48->96 ,67,120
        d2c = self.decode2conv(d2b)  # 1,96,48,64

        d1a = self.layer10(d2c)  # 1,96,135,240
        d1b = self.decode1(d1a, e1)  # 1,96+48->96,135,240
        d1c = self.decode1conv(d1b)  # 1,96,135,240
        # 其实在这里d1c的输出就要是49*3了，之后在这里改改
        d0a = self.layer11(d1c)  # 1,96,192,256
        d0b = self.decode0(d0a, final_input)  # 1,96+24->64,270,480
        d0c = self.decode0conv(d0b)  # 1,64,270,480

        # torch.cuda.synchronize()
        # time_en_e = time.perf_counter()
        # time_en = time_en_e - time_en_s
        # print('en/decoder:', time_en)

        # torch.cuda.synchronize()
        # time_dc1_s = time.perf_counter()

        dc1 = self.dconv1(torch.cat(
            (d0c,  # 1 64 270 480
             e1u),  # 1 48 270 480
            dim=1))  # 1 48 540 960

        # torch.cuda.synchronize()
        # time_dc1_e = time.perf_counter()
        # time_dc1 = time_dc1_e - time_dc1_s
        # print('time_dc1:',time_dc1)

        # torch.cuda.synchronize()
        # time_dc2_s = time.perf_counter()
        e1ux2=F.interpolate(e1u, scale_factor=2, mode='bilinear', align_corners=False)
        # print('dc1:',dc1.shape)
        # print('e1ux2:',e1ux2.shape)
        dc2 = self.dconv2(torch.cat(
            (dc1,  # 1 48 540 960
             e1ux2)  # 1 48 540 960
            , dim=1))  # 1 24 1080 1920
        # torch.cuda.synchronize()
        # time_dc2_e = time.perf_counter()
        # time_dc2 = time_dc2_e - time_dc2_s
        # print('time_dc2:', time_dc2)

        # torch.cuda.synchronize()
        # time_kernel_s = time.perf_counter()
        kernel1_weight = self.kernel1(dc2)  # dc2:1 32 768 1024 kernel:1 16*3*2+3 1080 1920
        kernel_noise = kernel1_weight[:, :27, :, :]  # 1 48 1080 1920
        kernel_warp = kernel1_weight[:, 27:54, :, :]  # 1 48 1080 1920
        weight_warp = kernel1_weight[:, 54:57, :, :]  # 1 3 1080 1920
        # weight_warp = normalize_0_1_batch(weight_warp)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        weight_warp = (torch.sin(weight_warp) + self.allOne) / 2
        weight_warp = weight_warp * mv_mask#1 3 1080 1920

        kernel2_weight = self.kernel2(dc1)  # dc1:1 48 540 960  ;  1,16*3+3 540 960
        kernel_noise1_2 = kernel2_weight[:, :27, :, :]  # 1 48  540 960
        weight_noise1_2 = kernel2_weight[:, 27:30, :, :]  # 1 3  540 960

        kernel3_weight = self.kernel3(d0c)  # d0c:1 64 270 480    ;  1 9*3+3 270 480
        kernel3_noise1_4 = kernel3_weight[:, :48, :, :]  # 1 48  270 480
        # kernel_warp = kernel3_weight[:, 48:96, :, :]  # 1 48  270 480
        weight_noise1_4 = kernel3_weight[:, 48:51 :, :]  # 1 3  270 480
        # weight_warp = kernel3_weight[:, 99:102, :, :]  # 1 3  270 480

        # torch.cuda.synchronize()
        # time_kernel_e = time.perf_counter()
        # time_kernel = time_kernel_e - time_kernel_s
        # print('kernel:', time_kernel)

        # torch.cuda.synchronize()
        # time_predict_s = time.perf_counter()
        noise_2 = self.down(noise).requires_grad_() # 1 3 540 960
        noise_4 = self.down(noise_2).requires_grad_()  # 1 3 1080 1920
        pred = self.kernel_pred(noise,kernel_noise)  # noisex4:1 3 1080 1920   kernel_noise:1 48 1080 1920->1 3 1080 1920
        pred2 = self.kernel_pred2(noise_2, kernel_noise1_2)  # 1 3 540 960 ; 1 48 540 960  ->  1 3 540 960
        pred3 = self.kernel_pred3(noise_4, kernel3_noise1_4)  # 1 3 270 540 ; 1 48 270 540 -> 1 3 270 540
        pred_p = self.kernel_predp(HR_Warp, kernel_warp)  # 1 3 1080 1920; 1 48 1080 1920 ->1 3 1080 1920




        # torch.cuda.synchronize()
        # time_predict_e = time.perf_counter()
        # time_predict = time_predict_e - time_predict_s
        # print('predict:', time_predict)

        # torch.cuda.synchronize()
        # time_contact_s = time.perf_counter()



        image_fine = pred2.requires_grad_()  # 1 3 540 960
        image_course = pred3.requires_grad_()  # 1 3 192 256
        Dif = self.downscale_23(image_fine).requires_grad_() # 1 3 192 256
        arphaDif = Dif.mul(weight_noise1_4).requires_grad_() # 1 3 192 256
        arphaic = image_course.mul(weight_noise1_4).requires_grad_()  # 1 3 192 256
        UarphaDif = self.upscale_nearest_23(arphaDif).requires_grad_() # 1 3 384 512
        Uarphaic = self.upscale_nearest_23(arphaic).requires_grad_() # 1 3 384 512
        halflerp = (image_fine - UarphaDif + Uarphaic).requires_grad_()  # 1 3 384 512

        # current = (self.allOne - weight_warp).mul(pred)  # 1 3 768 1024
        # histroy = weight_warp.mul(pred_p)  # 1 3 768 1024
        #
        # lerpResult = torch.add(current, histroy)  # 1 3 192 256



        image_fine_o = pred.requires_grad_()  # 1 3 1080 1920
        image_course_o = halflerp.requires_grad_()  # 1 3 384 512
        Dif_o = self.downscale_12(image_fine_o).requires_grad_()  # 1 3 384 512
        arphaDif_o = Dif_o.mul(weight_noise1_2).requires_grad_()  # 1 3 384 512
        arphaic_o = image_course_o.mul(weight_noise1_2).requires_grad_()  # 1 3 384 512
        UarphaDif_o = self.upscale_nearest_12(arphaDif_o).requires_grad_()  # 1 3 768 1024
        Uarphaic_o = self.upscale_nearest_12(arphaic_o).requires_grad_() # 1 3 768 1024
        lerpResult_spatial = (image_fine_o - UarphaDif_o + Uarphaic_o).requires_grad_() # 1 3 768 1024

        current = (self.allOne - weight_warp).mul(lerpResult_spatial).requires_grad_() # 1 3 768 1024
        histroy = weight_warp.mul(pred_p).requires_grad_()  # 1 3 768 1024

        lerpResult = torch.add(current, histroy).requires_grad_()  # 1 3 768 1024

        # torch.cuda.synchronize()
        # time_contact_e = time.perf_counter()
        # time_contact = time_contact_e - time_contact_s
        # print('contact:', time_contact)
        # print('in_all:',  time_en +time_dc1 + time_dc2+ time_kernel + time_predict + time_contact)
        # lerpResult_spatial = torch.abs(lerpResult_spatial)
        return lerpResult

            # ,noisex2,noisex4,pred,pred2,pred3,pred_p,halflerp,lerpResult_spatial,current,histroy,weight_warp
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

        core = core.view(batch_size, N, -1, color, height, width)

        #core=F.softmax(core,dim=2)
        core = torch.exp(core)
        dim_sum = core.sum(dim=2, keepdim=True)
        core = core /  dim_sum

        return core


    def forward(self, frames,core):
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
            color = 1
            N=1
            frames = frames.view(batch_size, N, color, height, width)#1,1,3,192,256


        core = self._convert_dict(core, batch_size, N, color, height, width)#1 1 25 3 192 256

        img_stack = []
        pred_img = []

        kernel = self.kernel_size
        if not img_stack:
            frame_pad = F.pad(frames, [kernel // 2, kernel // 2, kernel // 2, kernel // 2])#在输入图像周围加一圈宽2的像素

            for i in range(kernel):
                for j in range(kernel):

                    img_stack.append(frame_pad[..., i:i + height, j:j + width])

            img_stack = torch.stack(img_stack, dim=2)
        # print(img_stack.shape)

        pred_img.append(torch.sum(core.mul(img_stack), dim=2, keepdim=False))
        # print(pred_img.shape)
        pred_img = torch.stack(pred_img, dim=0)
        # print(pred_img.shape)
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)#N帧各自卷积
        pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)#N帧混合
        # print(pred_img.shape)
        # output=pred_img_i.view(batch_size , N*color , height , width)#1 6
        # pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
        return pred_img

class KernelConv1(nn.Module):
    """
    the class of computing prediction
    """
    def __init__(self, kernel_size=[5], sep_conv=False, core_bias=False):
        super(KernelConv1, self).__init__()
        self.kernel_size = sorted(kernel_size)
        self.sep_conv = sep_conv
        self.core_bias = core_bias

    def _sep_conv_core(self, core, batch_size, N, color, height, width):
        """
        convert the sep_conv core to conv2d core
        2p --> p^2
        :param core: shape: batch*(N*2*K)*height*width
        :return:
        """
        kernel_total = sum(self.kernel_size)
        core = core.view(batch_size, N, -1, color, height, width)
        if not self.core_bias:
            core_1, core_2 = torch.split(core, kernel_total, dim=2)
        else:
            core_1, core_2, core_3 = torch.split(core, kernel_total, dim=2)
        # output core
        core_out = {}
        cur = 0
        for K in self.kernel_size:
            t1 = core_1[:, :, cur:cur + K, ...].view(batch_size, N, K, 1, color, height, width)
            t2 = core_2[:, :, cur:cur + K, ...].view(batch_size, N, 1, K, color, height, width)
            core_out[K] = torch.einsum('ijklcno,ijlmcno->ijkmcno', [t1, t2]).view(batch_size, N, K * K, color, height, width)
            cur += K
        # it is a dict
        return core_out, None if not self.core_bias else core_3.squeeze()

    def _convert_dict(self, core, batch_size, N, color, height, width):
        """
        make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
        :param core: shape: batch_size*(N*K*K)*height*width
        :return: core_out, a dict
        """
        core_out = {}
        core = core.view(batch_size, N, -1, color, height, width)
        core_out[self.kernel_size[0]] = core[:, :, 0:self.kernel_size[0]**2, ...]
        bias = None if not self.core_bias else core[:, :, -1, ...]
        return core_out, bias

    def forward(self, frames, core, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        if self.sep_conv:
            core, bias = self._sep_conv_core(core, batch_size, N, color, height, width)
        else:
            core, bias = self._convert_dict(core, batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        # print("pred_img_i", pred_img_i.size())
        # N = 1
        pred_img_i_4 = torch.reshape(pred_img_i, (pred_img_i.size(0), pred_img_i.size(1), pred_img_i.size(3), pred_img_i.size(4)))  
        #print("pred_img_i", pred_img_i.size())
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i_4 += bias
        # print('white_level', white_level.size())
        pred_img_i_4 = pred_img_i_4 / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i_4
    
    def forward(self, frames, core, bias, white_level=1.0, rate=1):
        """
        compute the pred image according to core and frames
        :param frames: [batch_size, N, 3, height, width]
        :param core: [batch_size, N, dict(kernel), 3, height, width]
        :return:
        """
        if len(frames.size()) == 5:
            batch_size, N, color, height, width = frames.size()
        else:
            batch_size, N, height, width = frames.size()
            color = 1
            frames = frames.view(batch_size, N, color, height, width)
        img_stack = []
        pred_img = []
        kernel = self.kernel_size[::-1]
        for index, K in enumerate(kernel):
            if not img_stack:
                padding_num = (K//2) * rate
                frame_pad = F.pad(frames, [padding_num, padding_num, padding_num, padding_num])
                for i in range(0, K):
                    for j in range(0, K):
                        img_stack.append(frame_pad[..., i*rate:i*rate + height, j*rate:j*rate + width])
                img_stack = torch.stack(img_stack, dim=2)
            else:
                k_diff = (kernel[index - 1] - kernel[index]) // 2
                img_stack = img_stack[:, :, k_diff:-k_diff, ...]
            # print('img_stack:', img_stack.size())
            pred_img.append(torch.sum(
                core[K].mul(img_stack), dim=2, keepdim=False
            ))
        pred_img = torch.stack(pred_img, dim=0)
        # print('pred_stack:', pred_img.size())
        pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
        # print("pred_img_i", pred_img_i.size())
        # N = 1
        pred_img_i_4 = torch.reshape(pred_img_i, (pred_img_i.size(0), pred_img_i.size(1), pred_img_i.size(3), pred_img_i.size(4)))  
        #print("pred_img_i", pred_img_i.size())
        # if bias is permitted
        if self.core_bias:
            if bias is None:
                raise ValueError('The bias should not be None.')
            pred_img_i_4 += bias
        # print('white_level', white_level.size())
        pred_img_i_4 = pred_img_i_4 / white_level
        #pred_img = torch.mean(pred_img_i, dim=1, keepdim=True)
        # print('pred_img:', pred_img.size())
        # print('pred_img_i:', pred_img_i.size())
        return pred_img_i_4


class NSRRFeatureExtractionModel(BaseModel):
    """
    """

    def __init__(self):
        super(NSRRFeatureExtractionModel, self).__init__()
        kernel_size = 3
        # Adding padding here so that we do not lose width or height because of the convolutions.
        # The input and output must have the same image dimensions so that we may concatenate them
        padding = 1
        process_seq = nn.Sequential(
            # ModulatedDeformConvPack(in_channels=15, out_channels=16, kernel_size=(3, 3),
            #                          padding=(1, 1), stride=(1, 1), bias=True),
            # nn.ELU(),
            nn.Conv2d(15, 16, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # ModulatedDeformConvPack(in_channels=16, out_channels=16, kernel_size=(3, 3),
            #                          padding=(1, 1), stride=(1, 1), bias=True),
            # nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # ModulatedDeformConvPack(in_channels=16, out_channels=8, kernel_size=(3, 3),
            #                          padding=(1, 1), stride=(1, 1), bias=True),
            # nn.ELU(),
            nn.Conv2d(16, 8, kernel_size=kernel_size, padding=padding),
            nn.ReLU()
        )
        self.add_module("featuring", process_seq)

    def forward(self, input) -> torch.Tensor:
        # From each 3-channel image and 1-channel image, we construct a 4-channel input for our model.
        x_features = self.featuring(input)#1 8 270 480
        return x_features


class NSRRFeatureReweightingModel(BaseModel):
    """
    """

    def __init__(self):
        super(NSRRFeatureReweightingModel, self).__init__()
        # According to the paper, rescaling in [0, 10] after the final tanh activation
        # gives accurate enough results.
        self.scale = 10
        kernel_size = 3

        padding = 1
        # todo: I'm unsure about what to feed the module here, from the paper:

        process_seq = nn.Sequential(
            nn.Conv2d(23, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            Remap([-1, 1], [0, self.scale])
        )
        self.add_module("weighting", process_seq)

    def forward(self,
                current_features:#1 12 270 480
                    torch.Tensor,

                previous_features_warped:#1 12 270 480
                    torch.Tensor
                ) -> torch.Tensor:

        weighting_maps = self.weighting(torch.cat((current_features, previous_features_warped), dim=1))#1 1 270 480

        pre_features_reweighted = torch.mul(previous_features_warped , weighting_maps)#1 12 270 480

        # pre_features_reweighted = torch.zeros_like(previous_features_warped)
        # for i in range(weighting_maps.shape[0]):
        #     for j in range(weighting_maps.shape[1]):
        #         pre_features_reweighted[i*weighting_maps.shape[1] + j] = torch.mul(previous_features_warped[i*weighting_maps.shape[1] + j], weighting_maps[i, j])
        #

        return pre_features_reweighted

class Remap(BaseModel):
    """
    Basic layer for element-wise remapping of values from one range to another.
    """

    in_range: Tuple[float, float]
    out_range: Tuple[float, float]

    def __init__(self,
                 in_range: Union[Tuple[float, float], List[float]],
                 out_range: Union[Tuple[float, float], List[float]]
                 ):
        assert(len(in_range) == len(out_range) and len(in_range) == 2)
        super(BaseModel, self).__init__()
        self.in_range = tuple(in_range)
        self.out_range = tuple(out_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(
            torch.mul(torch.add(x, - self.in_range[0]), self.out_range[1] - self.out_range[0]),
            (self.in_range[1] - self.in_range[0]) + self.out_range[0])


class SuperResolutionBlock(nn.Module):
    """
    This class implements a super resolution block which is used after the original recurrent U-Net
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param final_output_channels: (int) Number of output channels for the mapping to image space
        """
        # Call super constructor
        super(SuperResolutionBlock, self).__init__()
        # Init layers
        self.layers = nn.Sequential(
            ModulatedDeformConvPack(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1), bias=True),
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),bias=True),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),#!!!!!!!!!!!!!!!!
            ModulatedDeformConvPack(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1), bias=True),
            # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),bias=True),
            nn.ELU(),
        )
        # Init residual mapping
        self.residual_mapping = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), padding=(0,0), stride=(1,1),bias=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)#!!!!!!!!!!!!!!!!!!!!!
        )

        # # Init output layer
        # self.output_layer = ModulatedDeformConvPack(in_channels=out_channels, out_channels=final_output_channels,
        #                                             kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
        #                                             bias=True) if final_block else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor)
        :return: (Tuple[torch.Tensor, torch.Tensor]) First, output tensor of main convolution. Second, image output
        """
        # Perform main layers
        output = self.layers(input)
        # Perform residual mapping
        output = output + self.residual_mapping(input)
        # Make image output
        # output = self.output_layer(output)
        return output



