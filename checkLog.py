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
# from torchsummary import summary
import torch.nn.functional as F

epoch = torch.load('./results/metrics/psnr.pt')

print(epoch)

# for name, param in epoch.named_parameters():
#     print(name, param.size())