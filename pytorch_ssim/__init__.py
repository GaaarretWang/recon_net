import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import torch.nn as nn
from utils import save_image

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, center_mask, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # conv1 = nn.Conv2d(1, 1, kernel_size=11, stride=1, padding=5, bias=False).to('cuda')
    # _1D_window = gaussian(11, 1.0).unsqueeze(1)
    # # _1D_window = torch.Tensor([1 for x in range(11)]).unsqueeze(1)
    # _1D_window /= _1D_window.sum()
    # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0).to('cuda')
    # conv1.weight = nn.Parameter(_2D_window)
    # loss_weight = conv1(conv1(center_mask))
    # loss_weight = torch.clamp(loss_weight, 0.3, 1)
    # save_image(
    #     loss_weight,
    #     fp='./ssim_tmp/loss_weight.exr',
    #     format='exr',
    #     nrow=1)

    # loss_weight_sum = torch.sum(loss_weight)
    # total = img1.shape[2] * img1.shape[3]
    # ratio = total / loss_weight_sum
    # loss_weight *= ratio

    ssim_map *= center_mask

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, center_mask):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, center_mask, self.size_average)


def ssim(img1, img2, center_mask, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, center_mask, size_average)
