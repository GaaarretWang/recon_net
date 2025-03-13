import os


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import deform_conv2d_onnx_exporter

from model import RecurrentUNet1
# from model import NSRRModel
from torchvision.ops.deform_conv import DeformConv2d

# from model import RecurrentUNet1
# from model import NSRRModel
# class Pre_Warp_RecurrentUNet(nn.Module):
#     """
#     This class implements a recurrent U-Net to perform super resolution base on the DeepFovea architecture
#     """

#     def __init__(self) -> None:
#         """
#         Constructor method
#         :param channels_encoding: (Tuple[Tuple[int, int]]) In and out channels in each encoding path
#         :param channels_decoding: (Tuple[Tuple[int, int]]) In and out channels in each decoding path
#         :param channels_super_resolution_blocks: (Tuple[Tuple[int, int]]) In and out channels in each s.r. block
#         """
#         # Call super constructor
#         super(Pre_Warp_RecurrentUNet, self).__init__()  # 调用了父类nn.Module的构造函数
#         input_size = 12

#         self.downlayer7 = nn.Sequential(
#             # ModulatedDeformConvPack(in_channels=4, out_channels=12, kernel_size=(3, 3),
#             #                          padding=(1, 1), stride=(1, 1), bias=True),
#             # # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),bias=True),
#             # nn.ELU(),
#             # ModulatedDeformConvPack(in_channels=12, out_channels=12, kernel_size=(3, 3),
#             #                          padding=(1, 1), stride=(1, 1), bias=True),
#             # # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1),bias=True),
#             # nn.ELU(),
#             nn.Conv2d(input_size, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#         )
#         self.downlayer7_down = nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0)

#         self.downlayer6 = nn.Sequential(
#             nn.Conv2d(16 + 4 * input_size, 24, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#             nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#         )
#         self.downlayer6_down = nn.Conv2d(24, 24, kernel_size=2, stride=2, padding=0)

#         self.downlayer5 = nn.Sequential(
#             nn.Conv2d(24 + 16 * input_size, 36, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#             nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#         )
#         self.downlayer5_down = nn.Conv2d(36, 36, kernel_size=2, stride=2, padding=0)

#         self.layer0 = nn.Sequential(
#             nn.Conv2d(36 + 64 * input_size, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#             nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),  # (1,24,720，1280)
#         )

#         self.uplayer5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#(270,480)
#         self.decode5conv = nn.Sequential(
#             nn.Conv2d(36 + 16 * input_size, 24, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(24, 24 + 9 + 3, kernel_size=3, stride=1, padding=1),
#             nn.ELU(),
#         )

#         self.uplayer6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#(270,480)
#         self.decode6conv = nn.Sequential(
#             nn.Conv2d(24 + 24 + 4 * input_size, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16 + 9 + 3, kernel_size=3, stride=1, padding=1),
#             nn.ELU(),
#         )

#         self.uplayer7 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False) 

#         self.gen_offset = nn.Conv2d(16 + 16 + input_size, 18, kernel_size=3, stride=1, padding=1)
#         self.offset = torch.zeros([1, 18, 1920, 1920])
#         self.input = torch.zeros([1, 44, 1920, 1920])
#         self.decode7conv = DeformConv2d(in_channels=16 + 16 + input_size, out_channels=14, kernel_size=(3, 3), padding=(1, 1))
#         self.elu = nn.ELU()

#         self.pixel_unshuffle_2 = nn.Sequential(
#             nn.PixelUnshuffle(2),
#             # nn.Conv2d(16, 12, kernel_size=1, stride=1, padding=0),
#         )
#         self.pixel_unshuffle_4 = nn.Sequential(
#             nn.PixelUnshuffle(4),
#             # nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
#         )
#         self.pixel_unshuffle_8 = nn.Sequential(
#             nn.PixelUnshuffle(8),
#             # nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
#         )
#         # self.pixel_shuffle_2 = nn.PixelShuffle(2)
#         # self.pixel_shuffle_4 = nn.PixelShuffle(4)


#     def forward(self, input):  # 1 24 192 256
#         """
#         Forward pass
#         :param input: (torch.Tensor) Input frame
#         :return: (torch.Tensor) Super resolution output frame
#         input:1 12 192 256
#         noise:1 3 192 256
#         warp:1 3 192 256:down_warp
#         HR_Warp:1 3 768 1024
#         warp_depth:1 1 192 256:mask_depth
#         """
#         unshuffle_2_input = self.pixel_unshuffle_2(input)
#         unshuffle_4_input = self.pixel_unshuffle_4(input)
#         unshuffle_8_input = self.pixel_unshuffle_8(input)

#         down7 = self.downlayer7(input)
#         down7_down = self.downlayer7_down(down7)

#         # down6 =  self.downlayer6(torch.cat((down7_down, unshuffle_2_input), dim=1))
#         # down6_down = self.downlayer6_down(down6)

#         # down5 = self.downlayer5(torch.cat((down6_down, unshuffle_4_input), dim=1))
#         # down5_down = self.downlayer5_down(down5)

#         # layer0 = self.layer0(torch.cat((down5_down, unshuffle_8_input), dim=1))

#         # up5 = self.uplayer5(layer0)
#         # decode5_output = self.decode5conv(torch.cat((down5, unshuffle_4_input), dim=1))

#         # up6 = self.uplayer6(decode5_output[:, :24, :, :])
#         # decode6_output = self.decode6conv(torch.cat((up6, down6, unshuffle_2_input), dim=1))

#         up7 = self.uplayer7(down7_down)
#         self.input = torch.cat((up7, down7, input), dim=1)
#         self.offset = self.gen_offset(self.input)
#         decode7_output = self.decode7conv(self.input, self.offset)
#         decode7_output1 = self.elu(decode7_output)


#         return decode7_output1

torch.cuda.amp.autocast()
deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()

generator_network = nn.DataParallel(RecurrentUNet1().cuda())
# generator_network = torch.load("./saved_data/k4-k3/generator_network_model_2.pt")
generator_network.eval()

dummy_input =(
    # torch.randn((1,2,1440,2560)).cuda(),
    # torch.randn((1,3,1440,2560)).cuda(),
    # torch.randn((1,3,1440,2560)).cuda(),
    torch.randn((1,3,1440,2560)).cuda(),
    torch.randn((1,3,1440,2560)).cuda(),
    torch.randn((1,1,1440,2560)).cuda(),
    torch.randn((1,1,1440,2560)).cuda(),
    torch.randn((1,1,1440,2560)).cuda(),
)
# deform_conv2d_onnx_exporter.register_deform_conv2d_onnx_op()
# model = Pre_Warp_RecurrentUNet()
# # input_names = ["input", "offset"]
# # output_names = ["output"]
# input_params = torch.zeros([1, 12, 1920, 1920])  # input
# # torch.onnx.export(model,
# #                   input_params,
# #                   "output.onnx",
# #                 #   input_names=input_names,
# #                 #   output_names=output_names,
# #                   opset_version=12)
with torch.no_grad():
    torch.onnx.export(generator_network.module,
                  dummy_input,
                  'generator_network_2.onnx',
                  verbose=True,
                  opset_version=14,
                #   dynamic_axes=None,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
print('finish')

#trtexec --onnx=generator_network_2.onnx --saveEngine=generator_network_2.trt --device=2 --fp16