import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from model import RecurrentUNet
# from model import NSRRModel
import numpy as np
import tqdm
input = torch.randn((1,12,192,256)).cuda()
noise = torch.randn((1,3,192,256)).cuda()
warp = torch.randn((1,3,192,256)).cuda()
warp_depth = torch.randn((1,1,192,256)).cuda()
HRwarp = torch.randn((1,3,768,1024)).cuda()
repetitions = 300
pre_radiance = torch.randn((2,3,192,256)).cuda()
predepth = torch.randn((2,1,192,256)).cuda()
mv = torch.randn((2,2,192,256)).cuda()
prediction_lr = torch.randn((1,3,192,256)).cuda()

mask =  torch.randn((768,1024)).cuda()

generator_network = nn.DataParallel(RecurrentUNet().cuda())
generator_network.eval()

import time
# Create CUDA events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
outputs = generator_network(input,noise,warp,HRwarp,warp_depth,mask)
# Run inference with timing
sum = 0
for i in range(300):

    torch.cuda.synchronize()
    clock0 = time.perf_counter()
    outputs = generator_network(input,noise,warp,HRwarp,warp_depth,mask)
    torch.cuda.synchronize()
    clock1 = time.perf_counter()
      # Synchronize CUDA context
    inference_time = clock1 - clock0
    print(f"Inference time: ", inference_time)
    sum = sum+inference_time
alltime = sum / 300
print(f"all time: ",alltime)