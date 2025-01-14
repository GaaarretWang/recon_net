from argparse import ArgumentParser
import os

# Manage command line arguments
parser = ArgumentParser()
parser.add_argument("--train", default=False, action="store_true",
                    help="Binary flag. If set training will be performed.")
parser.add_argument("--trainD", default=False, action="store_true",
                    help="Binary flag. If set training will be performed.")
parser.add_argument("--val", default=False, action="store_true",
                    help="Binary flag. If set validation will be performed.")
parser.add_argument("--valD", default=False, action="store_true",
                    help="Binary flag. If set validation will be performed.")
parser.add_argument("--test", default=False, action="store_true",
                    help="Binary flag. If set testing will be performed.")
parser.add_argument("--inference", default=False, action="store_true",
                    help="Binary flag. If set inference will be performed.")
parser.add_argument("--inference_data", default="./REDS/val/val_sharp", type=str,
                    help="Path to inference data to be loaded.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If multi GPU training should be utilized set flag.")
parser.add_argument("--load_model", default="", type=str,
# parser.add_argument("--load_model", default="./saved_data/generator_network_model_25.pt", type=str,
# parser.add_argument("--load_model", default="./saved_data/tmp.pt", type=str,
                    help="Path to model to be loaded.")
# the number of input features.
parser.add_argument('-f', '--featureNum', default=1252, type=int,
                    help='the number of input features (default: 1252)')    
# the parameters for the net.
parser.add_argument('--seqLength', default=50, type=int,
                    help=' the length of the head sequence (default: 50)')        
parser.add_argument('--seqFeatureNum', default=2, type=int,
                    help=' the number of features in the head sequence (default: 2)')        
parser.add_argument('--saliencyWidth', default=24, type=int,
                    help='the width/height of the saliency map (default: 24)')    
parser.add_argument('--saliencyNum', default=2, type=int,
                    help='the number of the input saliency maps(default: 2)')                     
parser.add_argument('--n_output', default=2, type=int,
                    help='the number of the input saliency maps(default: 2)')                     
# the dropout rate of the model.
parser.add_argument('--dropout_rate', default=0.5, type=float,
                    help='the dropout rate of the model (default: 0.5)')       
parser.add_argument('--details', default="", type=str,
                    help='save the training details')
parser.add_argument("--cuda_devices", default="1", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")


# Get arguments
args = parser.parse_args()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RecurrentUNet1
# from model import NSRRModel
# from discriminator import Discriminator, FFTDiscriminator
from vgg_19 import VGG19
from model_wrapper import ModelWrapper
from dataset import REDS, REDSFovea, REDSParallel, REDSFoveaParallel, reds_parallel_collate_fn
from models.DGazeModels import DGaze_ET
from models.weight_init import weight_init
import misc
# from ranger21 import Ranger21
from ranger import Ranger
from l1loss import L1Loss
from lpips_pytorch import LPIPS

if __name__ == '__main__':
    epochs = 60
    width = 2560
    height = 1440
    batch_size = 3
    # Init networks
    generator_network = nn.DataParallel(RecurrentUNet1().cuda())
    if args.load_model != "":
        generator_network = torch.load(args.load_model)
    print("1111")

    # vgg_19 = nn.DataParallel(VGG19().cuda())
    L1Loss = nn.DataParallel(L1Loss().cuda())
    lpips = nn.DataParallel(LPIPS(
        net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
        version='0.1'  # Currently, v0.1 is supported
    ).cuda())

    print("2222")

    # generator_network_optimizer = torch.optim.Adam(generator_network.parameters() , lr=0.0001)
    # generator_network_optimizer = Ranger21(generator_network.parameters(), lr=1.25e-3, weight_decay=0.01, num_epochs=20, num_batches_per_epoch=2700)
    # generator_network_optimizer = Ranger(generator_network.parameters(), lr=5e-5, weight_decay=0.01)
    generator_network_optimizer = Ranger(generator_network.parameters(), lr=1e-3, weight_decay=0.01)
    # Dmv_network_optimizer = torch.optim.Adam(Dmv_network.parameters(), lr=0.0001)
    # Tmv_network_optimizer = torch.optim.Adam(Tmv_network.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_network_optimizer, 30, 1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(generator_network_optimizer, 10, gamma=0.5, last_epoch=-1)

    # Init model wrapper
    print("3333")
    model_wrapper = ModelWrapper(generator_network=generator_network,
                                 # Dmv_network=Dmv_network,
                                 # Tmv_network=Tmv_network,
                                #  vgg_19=vgg_19,
                                 L1Loss = L1Loss,
                                 lpips = lpips,
                                 generator_network_optimizer=generator_network_optimizer,
                                 scheduler = scheduler,
                                 details = args.details,
                                 batch_size = batch_size,
                                #  training_dataloader=None, 
                                 training_dataloader=DataLoader(
                                    # Bistro_faster1 Square_fast
                                    #  REDSFovea(path='./pica/train/train_sharp', batch_size=1,
                                    #  REDSFovea(path='../Bistro_low/train', batch_size=1,
                                     REDSFovea(path='/home/wgy/files/DGaze_and_Recon/new_image_test/image/Dataset_2/train'),
                                               batch_size=batch_size, shuffle=True, num_workers=20),  # b=2
                                #  validation_dataloader=None, 
                                 validation_dataloader=DataLoader(
                                    #  REDSFovea(path='./pica/val/val_sharp', batch_size=1,
                                    #  REDSFovea(path='../Bistro_low/valid', batch_size=1,
                                     REDSFovea(path='/home/wgy/files/DGaze_and_Recon/new_image_test/image/Dataset_1/valid'),
                                               batch_size=1, shuffle=False,num_workers=20),
                                 validation_dataloader_1=DataLoader(
                                    #  REDSFovea(path='./pica/val/val_sharp', batch_size=1,
                                    #  REDSFovea(path='../Bistro_low/valid', batch_size=1,
                                     REDSFovea(path='/home/wgy/files/DGaze_and_Recon/new_image_test/image/Dataset_2/valid'),
                                               batch_size=1, shuffle=False,num_workers=20),
                                 test_dataloader=None, 
                                #  test_dataloader=DataLoader(
                                #     #  REDSFovea(path='./pica/val/val_sharp', batch_size=1,
                                #     #  REDSFovea(path='../Bistro_low/valid', batch_size=1,
                                #      REDSFovea(path='/home/wgy/files/DGaze_and_Recon/high_fmv/tmp/cuda-cpp_wrapper_example/build/output', batch_size=1),
                                #                batch_size=1, shuffle=False,num_workers=1),
                                 width = width, height = height)


    # Perform training
    if args.train:
        print('begin train')
        model_wrapper.train(epochs=epochs)#20
    # Perform final validation
    if args.val:
        model_wrapper.validate(epoch = 0, plot_after_n_iterations = 10)
        model_wrapper.validate_1(epoch = 0, plot_after_n_iterations = 10)
    # Perform testing
    if args.test:
        model_wrapper.test()
    # Perform validation
    if args.inference:
        # Load data
        inference_data = misc.load_inference_data(args.inference_data)
        model_wrapper.inference(inference_data)