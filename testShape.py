from argparse import ArgumentParser
import os



import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RecurrentUNet
from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from pwc_net import PWCNet
from model_wrapper import ModelWrapper
from dataset import REDS, REDSFovea, REDSParallel, REDSFoveaParallel, reds_parallel_collate_fn
from lossfunction import AdaptiveRobustLoss
from resample.resample2d import Resample2d
import misc
from torchsummary import summary
import torch.nn.functional as F

validation_dataloader=DataLoader(REDSFovea(path='./REDS/val/val_sharp'), batch_size=1, shuffle=False,
                                     num_workers=0)
train_dataloader=DataLoader(REDSFovea(path='./REDS/train/train_sharp'), batch_size=1, shuffle=False,
                                     num_workers=0)
print('len:',len(validation_dataloader))
print('train len:' ,len(train_dataloader))

generator_network = nn.DataParallel(RecurrentUNet().cuda())
vgg_19 = nn.DataParallel(VGG19().cuda())
discriminator_network = nn.DataParallel(Discriminator().cuda())
fft_discriminator_network = nn.DataParallel(FFTDiscriminator().cuda())
pwc_net = nn.DataParallel(PWCNet().cuda())#输入必须两帧，以提取光流
resample = nn.DataParallel(Resample2d().cuda())#
# summary(generator_network,(18,192,256))
# generator_network.load_state_dict(torch.load("./results/2023-04-02-parallel/models/generator_network_30.pt"),strict=False)
# input_tensor = torch.randn((18,768,1024))
# summary(discriminator_network, (18,768,1024))
'''print('---------------------------------------------------------')
import dataset
import matplotlib.pyplot as plt
dataset = dataset.REDS()
pwc_net = PWCNet().cuda().eval()
images = dataset[0][1]#18,768,1024
image_1 = images[:3].unsqueeze(dim=0).cuda()#1,3,768,1024
image_2 = images[3:6].unsqueeze(dim=0).cuda()#1,3,768,1024
prediction_pair = torch.cat((image_1, image_2),
                                dim=1)  # [:-1,:,:,:]的意思，即第一个维度取前5.所以是取了前五张和后五张
print('images:',images.shape)
print('image_1:',image_1.shape)
print('image_2:',image_2.shape)
print('prediction_pair:',prediction_pair.shape)
plt.imshow(image_1[0].detach().cpu().numpy().transpose(1, 2, 0))
plt.show()
plt.imshow(image_2[0].detach().cpu().numpy().transpose(1, 2, 0))
plt.show()


flow = pwc_net(prediction_pair)

plt.imshow(flow.cpu().detach().numpy()[0, 0])
plt.show()
plt.imshow(flow.cpu().detach().numpy()[0, 1])
plt.show()
image_rec = resample(image_2, flow)

print(image_rec.shape)

plt.imshow(image_rec[0].detach().cpu().numpy().transpose(1, 2, 0))
plt.show()
print('---------------------------------------------------------')'''
# for index_sequence, batch in enumerate(validation_dataloader):  # 遍历每个序列
#     # Unpack batch
#     input, label, new_sequence = batch  # 一个序列,18,h,w
#     print('input:',input.shape)
#     print('label:', label.shape)
#
#     prediction = generator_network(input.detach())  # b c h w=1,18,h,w
#     print('predict:',prediction.shape)
#     prediction_reshaped_4d = prediction.reshape(prediction.shape[0] * (prediction.shape[1] // 3), 3,
#                                                 prediction.shape[2], prediction.shape[3])  # b f c h w
#     print('prediction_reshaped_4d:', prediction_reshaped_4d.shape)
#     label_reshaped_4d = label.reshape(label.shape[0] * (label.shape[1] // 3), 3, label.shape[2],
#                                       label.shape[3])
#     print('label_reshaped_4d:', label_reshaped_4d.shape)
#     #
#     # avg=F.avg_pool2d(prediction_reshaped_4d, kernel_size=2)
#     # print('avg:',avg.shape)
#     #
#     dis_label=discriminator_network(label)
#     print('dis_label:',dis_label.shape)
#     #
#     dis_predict = discriminator_network(prediction.detach())
#     print('dis_predict:', dis_predict.shape)
#     #
#     prediction = generator_network(input.detach())
#     # # Reshape prediction and label for vgg19
#     # prediction_reshaped_4d = prediction.reshape(prediction.shape[0] * (prediction.shape[1] // 3), 3,
#     #                                             prediction.shape[2], prediction.shape[3])
#     # print('prediction_reshaped_4d:', prediction_reshaped_4d.shape)
#     pre1 = prediction_reshaped_4d[:-1]
#     print('pre1:',pre1.shape)
#     pre2=prediction_reshaped_4d[1:]
#     print('pre2:', pre2.shape)
#     prediction_pair = torch.cat((pre1 , pre2),dim=1)
#     print('prediction_pair:', prediction_pair.shape)
#     #
#     fft_label = fft_discriminator_network(label)
#     print('fft_label:', fft_label.shape)
#     #
#     input = label.view(2, 3, label.shape[2], label.shape[3])#6,3,728,1024
#     #
#     '''red_in = fft_input[:, 0, None]#6,1,728,1024 只取r通道
#     print('red_in:' , red_in.shape)
#     red_in_p = red_in.permute([0, 2, 3, 1])#6 728 1024 1
#     print('red_in_p:', red_in_p.shape)
#
#     red_fft = torch.fft.rfft(red_in_p)
#     print('red_fft:', red_fft.shape)#6,728,1024,1
#
#     red_fft_r = torch.stack((red_fft.real, red_fft.imag), -1)
#     print('red_fft_r:', red_fft_r.shape)  # 6,728,1024,1,2
#     red_fft_features = red_fft_r.permute([0, 3, 1, 2, 4])
#     print('red_fft_features:', red_fft_features.shape)  # 6, 1, 768, 1024, 2:frames, in channels, height, width, real + imag'''
#
#     red_in = input[:, 0, None].permute([0, 2, 3, 1])  # f h w c
#     green_in = input[:, 1, None].permute([0, 2, 3, 1])
#     blue_in = input[:, 2, None].permute([0, 2, 3, 1])
#
#     red_fft = torch.fft.rfft(red_in)
#     red_fft_r = torch.stack((red_fft.real, red_fft.imag), -1)
#     red_fft_features = red_fft_r.permute([0, 3, 1, 2, 4])
#
#     green_fft = torch.fft.rfft(green_in)
#     green_fft_r = torch.stack((green_fft.real, green_fft.imag), -1)
#     green_fft_features = green_fft_r.permute([0, 3, 1, 2, 4])
#
#     blue_fft = torch.fft.rfft(blue_in)
#     blue_fft_r = torch.stack((blue_fft.real, blue_fft.imag), -1)
#     blue_fft_features = blue_fft_r.permute([0, 3, 1, 2, 4])
#
#     print('red_fft_features:', red_fft_features.shape)
#     print('green_fft_features:', green_fft_features.shape)
#     print('blue_fft_features:', blue_fft_features.shape)
#     #
#     output = torch.cat((red_fft_features, green_fft_features, blue_fft_features), dim=1).permute(2, 3, 0, 1, 4)#h w f c r+i
#     print('output:', output.shape)
#     #
#     output = output.contiguous().view(output.shape[0], output.shape[1], -1).permute(2, 0, 1).unsqueeze(dim=0)  #先保留h w维，把后面三维展平，再变成c h w，再添加b，bchw
#     print('output:', output.shape)#1，36，768，1024
#     #
#     # output = torch.randn((1,1,6,6,8))
#     # avg_output = F.adaptive_avg_pool3d(input=output, output_size=(1, 16, 16))
#     # print('avg:',avg_output.shape)
#     # #
#     # flatten = avg_output.flatten(start_dim=1)
#     # print('flatten:', flatten.shape)
#     #
#     # final_linear = nn.utils.spectral_norm(nn.Linear(in_features=256, out_features=1, bias=True))
#     # output = final_linear(flatten)
#     # print('output:',output.shape)
#     #
#     input, label, new_sequence = batch
#     prediction = generator_network(input.detach())
#     fft_pre = fft_discriminator_network(prediction.detach())
#     print('fft_pre:', fft_pre.shape)
    #
    #
    #
