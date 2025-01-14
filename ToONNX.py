import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RecurrentUNet1
from model import DMVModel
from model import TMVModel
from model import DepthModel
# from model import NSRRModel
from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from pwc_net import PWCNet
from model_wrapper import ModelWrapper
from dataset import REDS, REDSFovea, REDSParallel, REDSFoveaParallel, reds_parallel_collate_fn
from lossfunction import AdaptiveRobustLoss
from resample.resample2d import Resample2d
import misc



generator_network = nn.DataParallel(RecurrentUNet1().cuda())
# generator_network = torch.load("./saved_data/k4-k3/generator_network_model_2.pt")

# print(generator_network)

generator_network.eval()

dummy_input =(
    torch.randn((1,2,1080,1920)).cuda(),
    torch.randn((1,1,1080,1920)).cuda(),
    torch.randn((1,1,1080,1920)).cuda(),
    torch.randn((1,3,1080,1920)).cuda(), 
    torch.randn((1,1,1080,1920)).cuda(),
    torch.randn((1,3,1080,1920)).cuda(),
    torch.randn((1,1,1080,1920)).cuda()
)



torch.onnx.export(generator_network.module,
                  dummy_input,
                  'generator_network_2.onnx',
                  verbose=True,
                  opset_version=11,
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
print('finish')