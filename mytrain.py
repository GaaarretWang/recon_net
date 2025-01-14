import os
import random
import numpy as np
import torch
from torchvision.utils import save_image
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from PIL import Image
#from matplotlib import pyplot as plt
from datetime import datetime,timedelta
from MyDatasetNew import *
from MyNetNew import MyNetNew
from MyNetNew import sRGBGamma
from MyNetNew import LossFunc
from MyNetNew import millis
is_gpu = torch.cuda.is_available()
gpu_nums = torch.cuda.device_count()
gpu_index = torch.cuda.current_device()
print(is_gpu,gpu_nums,gpu_index)
device_name = torch.cuda.get_device_name(gpu_index)
print(device_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================ step 1/5 数据 ============================
#BASE_DIR = "./mysequence/animeRobot/"
BASE_DIR = "F:/NeuralTemporalKP/"
MODEL_NAME = "MyNet_softMax"
MODEL_SAVE = "F:/NeuralTemporalKP/saveModel_revise/MyNet_softMax/"
CHECK_POINT ="F:/NeuralTemporalKP/checkPoint_revise/MyNet_softMax/"
TEST_RESULT = "F:/NeuralTemporalKP/dataSet/robot/right(3-10-17-24-31-38-45)/"
#构建数据集实例
train_data = MyDatasetNew("F:/NeuralTemporalKP/trainSet/",1420,False)

train_dataset_size = len(train_data)

test_data = MyDatasetNew("F:/NeuralTemporalKP/testSet/",1390,True)
#test_data = MyDatasetNew("F:/NeuralTemporalKP/dataSet/robot/right(3-10-17-24-31-38-45)/",530,True)
test_dataset_size = len(test_data)

print('The number of training noise images = %d' % train_dataset_size)
print('The number of testing noise images = %d' % test_dataset_size)

#构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
# ============================ step 2/5 模型 ============================
net = MyNetNew().to(device)
#训练新的网络
net.initialize_weights()

#加载已有网络
#LOAD_DIR = "F:/NeuralTemporalKP/saveModel/MyNet(new)/MyNet(new)_205.pth"
#net.load_state_dict(torch.load(LOAD_DIR))

net.print_networks()
# ============================ step 3/5 损失函数 ============================

loss_L1 = nn.L1Loss().to(device)

#kernelPredict的损失函数？？？
loss_func = LossFunc(
        coeff_basic=1.0,
        coeff_anneal=1.0,
        gradient_L1=True,
        alpha=0.9998,
        beta=100.0
    )

# ============================ step 4/5 优化器 ============================
optimizer = optim.Adam(net.parameters(), lr=0.0001)                        # 选择优化器
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)     # 设置学习率下降策略

# ============================ step 5/5 训练 ============================
total_iters = 0
train_curve = list()

# 构建 SummaryWriter
#log_dir = "./directPredict_log/train_log"
#writer = SummaryWriter(log_dir=log_dir, comment='directPredict', filename_suffix="training")


for epoch in range(0,200):
    loss_mean = 0.
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    #if (epoch == 50 or epoch == 100 or epoch == 150 or epoch == 200 or epoch == 250 or epoch == 300):
    if (epoch == 200):#永远达不到，这里主要是为了懒得删掉而搞得
        #average_data =0
        #average_ext = 0
        #average_pred = 0
        #average_lerp = 0
        #average_warp = 0
    #if(epoch==50 or epoch==100 or epoch==150 or epoch==200 or epoch==250 or epoch==300 or epoch==350 or epoch==400 or epoch==450 or epoch==499):
        for i, data in enumerate(test_loader):  # 对于单个epoch的每个iter遍历
            start_getdata_time = datetime.now()

            noise = data['noise']
            input = data['input']
            warpPrev = data['warpPrev']
            prevRef = data['prevRef']
            prevDenoise = data['prevDenoise']

            reference = data['reference']
            NmotionVector = data['NmotionVector']
            NmotionVector.detach()

            traMV = data['traMV']
            occMV = data['occMV']
            preTraMV1 = data['preTraMV1']
            preOccMV1 = data['preOccMV1']
            preTraMV2 = data['preTraMV2']
            preOccMV2 = data['preOccMV2']

            get_data_time = millis(start_getdata_time)

            # denoise = net(input)#directpredict和directGbuffer
            #current,histroy,pred= net(input, noise,warpPrev)
            current, histroy, pred = net(input, noise, warpPrev)
            #if(i>=1):
                #average_data +=get_data_time
                #average_ext +=extract_time
                #average_pred+=predict_time
                #average_lerp+=lerp_time
            test_data.prevDenoise_tensor = pred.squeeze(0)

            #net.save_pictures(epoch, i, warpPrev, 'warpPrev', TEST_RESULT+"%d/warpPrev/"%epoch)
            #net.save_pictures(epoch, i, warpPrev, 'warpPrev', TEST_RESULT + "warpPrev/")

            save_image(warpPrev, TEST_RESULT + "Ours/noclip/warpPrev/%04d" % i +'.png')

            cu = pred.detach()

            #start_warp_time = datetime.now()

            #warpPrev,warp_time = test_data.tp(cu, NmotionVector, reference)
            warpPrev, warp_time = train_data.tp(cu, NmotionVector, reference)
            #warp_time = millis(start_warp_time)
            # if (i % 10 == 0):
            #     print("iter:{:.4f}, data::{:.4f}, extract:{:.4f}, predict:{:.4f}, lerp:{:.4f}， warp:{:.4f}".format(i, get_data_time,extract_time,
            #                                                                                          predict_time,
            #                                                                                          lerp_time,
            #                                                                                          warp_time))
            # if (i >= 1):
            #     average_warp +=warp_time

            test_data.warpPrev_tensor = warpPrev.squeeze(0)  # 把当前获得的warpPrev传回数据集的warpPrev_tensor以作为输入

            prevDenoise = prevDenoise.detach()

            loss_spatial, loss_temporal,loss_mask= loss_func(sRGBGamma(pred), sRGBGamma(reference),
                                                                 sRGBGamma(prevRef), sRGBGamma(prevDenoise),
                                                                 sRGBGamma(noise),
                                                                 traMV, occMV, preTraMV1, preOccMV1, preTraMV2,
                                                                 preOccMV2)
            if(epoch>=100):
                loss = loss_spatial + loss_temporal+loss_mask
            else:
                loss = loss_spatial + loss_temporal
            #print(loss_mask)
            loss_mean += loss.item()

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            torch.cuda.memory_summary(device=None, abbreviated=False)

            loss_mean = 0.
            #net.save_pictures(epoch, i, pred, 'denoise', TEST_RESULT+"%d/"%epoch)
            #net.save_pictures(epoch, i, pred, 'denoise', TEST_RESULT+"denoise/")

            save_image(pred, TEST_RESULT + "Ours/noclip/denoise/%04d" % i + '.png')

            total_iters += 1
            epoch_iter += 1

        #net.save_networks(epoch, MODEL_NAME, MODEL_SAVE)
        # average_ext = average_ext / 1389
        # average_pred = average_pred / 1389
        # average_lerp = average_lerp / 1389
        # average_warp =average_warp/1389
        # average_data = average_data/1390
        # print(
        # "average_data:{:.4f},average_ext:{:.4f}, average_pred:{:.4f}, average_lerp:{:.4f}, average_warp:{:.4f}".format(average_data,average_ext, average_pred, average_lerp,average_warp))
        continue
    net.save_networks(epoch, MODEL_NAME, MODEL_SAVE)

    net.train()
    for i, data in enumerate(train_loader):#对于单个epoch的每个iter遍历
        iter_start_time = time.time()#本次iter开始的时间

        #print("iter:",i)

        #前向传播
        noise= data['noise']
        input= data['input']
        warpPrev= data['warpPrev']
        prevRef = data['prevRef']
        prevDenoise = data['prevDenoise']

        reference = data['reference']
        NmotionVector = data['NmotionVector']
        NmotionVector.detach()


        traMV = data['traMV']
        occMV =data['occMV']
        preTraMV1 = data['preTraMV1']
        preOccMV1 = data['preOccMV1']
        preTraMV2 = data['preTraMV2']
        preOccMV2 = data['preOccMV2']

        #denoise = net(input)#directpredict和directGbuffer
        current,histroy,pred= net(input,noise,warpPrev)


        if(i%10!=9):
            train_data.prevDenoise_tensor =pred.squeeze(0)

        if i == 418 or i == 419 or i == 688 or i == 689 or i == 908 or i == 909 or i == 1098 or i == 1099:
        #if i == 218 or i == 219 or i == 408 or i == 409:
            net.save_pictures(epoch, i, warpPrev, 'warpPrev', CHECK_POINT)
        #if epoch ==49 or epoch ==99 or epoch ==149 or epoch ==199 or epoch ==249 or epoch ==299 or epoch ==349 or epoch ==399 or epoch ==449 or epoch ==498:
        if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch == 299:
            net.save_pictures(epoch, i, warpPrev, 'warpPrev', TEST_RESULT + "%d/warpPrev/" % epoch)
        cu=pred.detach()
        #warpPrev,warp_time = train_data.tp(cu, NmotionVector, reference)
        warpPrev, warp_time = train_data.tp(cu, NmotionVector, reference)
        train_data.warpPrev_tensor = warpPrev.squeeze(0)  # 把当前获得的warpPrev传回数据集的warpPrev_tensor以作为输入

        prevDenoise = prevDenoise.detach()

        loss_spatial, loss_temporal,loss_mask= loss_func(sRGBGamma(pred), sRGBGamma(reference),
                                                             sRGBGamma(prevRef), sRGBGamma(prevDenoise),
                                                             sRGBGamma(noise),
                                                             traMV, occMV, preTraMV1, preOccMV1, preTraMV2,
                                                             preOccMV2)


        #后向传播
        if (i % 10 != 9):
            optimizer.zero_grad()

        if(epoch<100):
            loss_mask=0

        loss = loss_spatial + loss_temporal +loss_mask


        if (i % 10 != 9):
            loss.backward(retain_graph=False)  # 添加retain_graph=True标识，让计算图不被立即释放
            # 更新权重
            optimizer.step()

        loss_mean += loss.item()
        train_curve.append(loss.item())

        iter_data_time = time.time()

        #net.train()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

        #每10个iter都打印log信息，并在tensorboard更新图片和loss
        if i % 10 == 8 or i % 10 == 9:
            #打印信息
            loss_mean = loss_mean / 10
            t_data = iter_start_time - iter_data_time
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] loss_spatial: {:.4f}  loss_temporal: {:.4f} loss_maskL1: {:.4f}  loss: {:.4f}   Time: {:.2f}sec".format(
            epoch, 300, i, len(train_loader), loss_spatial,loss_temporal,loss_mask,loss,time.time() - epoch_start_time))

        loss_mean = 0.

        if i == 418 or i== 419 or i==688 or i==689 or i==908 or i==909 or i==1098 or i==1099:
            net.save_pictures(epoch, i, pred, 'denoise', CHECK_POINT)

        if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch == 299:
        #if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch == 299 or epoch == 349 or epoch == 399 or epoch == 449 or epoch == 498:
            net.save_pictures(epoch, i, pred, 'denoise', TEST_RESULT+"%d/"%epoch)

        total_iters += 1
        epoch_iter +=1

    print('epoch结束 %d / %d \t 花费时间: %d sec' % (epoch, 300, time.time() - epoch_start_time))

