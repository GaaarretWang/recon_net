import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from model import RecurrentUNet1
# from model import NSRRModel
import numpy as np
import tqdm


generator_network = nn.DataParallel(RecurrentUNet1().cuda())
# nsrr_network = nn.DataParallel(NSRRModel(scale_factor=4, number_previous_frames=2, batch_size=1, width=256, height=192).cuda())
# generator_network = torch.load("./results/L1+tem/models/generator_network_model_20.pt")
# nsrr_network = torch.load("./results/20230518-NSRR/models/nsrr_network_0.pt")
# nsrr_network.to('cuda')
# nsrr_network.eval()
generator_network.eval()
#方法一：知乎
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
#
# print('warm up ...\n')
# with torch.no_grad():
#     for _ in range(100):
#         _ = generator_network(input,noise,warp,HRwarp,warp_depth,mask)
#
# torch.cuda.synchronize()
#
# timings = np.zeros((repetitions, 1))
#
# print('testing ...\n')
# with torch.no_grad():
#     for rep in tqdm.tqdm(range(repetitions)):
#         start_event.record()
#         _ = generator_network(input,noise,warp,HRwarp,warp_depth,mask)
#         end_event.record()
#         torch.cuda.synchronize() # 等待GPU任务完成
#         curr_time = start_event.elapsed_time(end_event) # 从 starter 到 ender 之间用时,单位为毫秒
#         timings[rep] = curr_time
#
# avg = timings.sum()/repetitions
# print('\navg={}\n'.format(avg))


'''#方法一
#---------------------------超分
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        _ = nsrr_network(pre_radiance, predepth, mv, prediction_lr,depth)
torch.cuda.synchronize()
timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        start_event.record()
        _ = nsrr_network(pre_radiance, predepth, mv, prediction_lr,depth)
        end_event.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = start_event.elapsed_time(end_event) # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time
avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))'''




'''#方法二：
timer = benchmark.Timer(
    stmt="generator_network(input,noise,warp)",  # 测量的语句
    setup="from __main__ import generator_network, input , noise , warp",  # 执行语句前的设置
    num_threads=1
)
print(timer.timeit(300))'''


'''timer = benchmark.Timer(
    stmt="nsrr_network(pre_radiance, predepth, mv, prediction_lr,depth)",  # 测量的语句
    setup="from __main__ import nsrr_network, pre_radiance, predepth, mv, prediction_lr,depth",  # 执行语句前的设置
    num_threads=1
)
print(timer.timeit(300))'''

# depth_network = torch.load("./results/reconmv/models/depth_model_30.pt")
# depth =  torch.randn((1,1,192,256)).cuda()
# mask = torch.randn((1,1,192,256)).cuda()
#方法三：
# import time
# # Create CUDA events for timing
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# outputs = depth_network(depth,mask)
# # Run inference with timing
# sum = 0
# for i in range(300):
#
#     torch.cuda.synchronize()
#     clock0 = time.perf_counter()
#     outputs = outputs = depth_network(depth,mask)
#     torch.cuda.synchronize()
#     clock1 = time.perf_counter()
#       # Synchronize CUDA context
#     inference_time = clock1 - clock0
#     print(f"Inference time: ", inference_time)
#     sum = sum+inference_time
# alltime = sum / 300
# print(f"all time: ",alltime)
#

mask_img = torch.randn((1,3,1080,1920)).cuda()
recon_results = torch.randn((1,3,1080,1920)).cuda()
grid_sub = torch.randn((1,2,1080,1920)).cuda()
mask_normal = torch.randn((1,3,270,480)).cuda()
mask_depth = torch.randn((1,1,270,480)).cuda()
pre_warp_color = torch.randn((1,3,1080,1920)).cuda()
sample_time = torch.randn((1,1,1080,1920)).cuda()
samewarp = torch.randn((1,3,1080,1920)).cuda()
mv_masks = torch.randn((1,1,1080,1920)).cuda()
# input = torch.randn((1,12,192,256)).cuda()
# noise = torch.randn((1,3,192,256)).cuda()
# warp = torch.randn((1,3,192,256)).cuda()
# warp_depth = torch.randn((1,1,192,256)).cuda()
# HRwarp = torch.randn((1,3,768,1024)).cuda()
# repetitions = 300
# pre_radiance = torch.randn((2,3,192,256)).cuda()
# predepth = torch.randn((2,1,192,256)).cuda()
# mv = torch.randn((2,2,192,256)).cuda()
# prediction_lr = torch.randn((1,3,192,256)).cuda()

# mask =  torch.randn((1,1,768,1024)).cuda()
import time
# Create CUDA events for timing
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# outputs = generator_network(input,noise,warp,HRwarp,warp_depth,mask)
# Run inference with timing
torch.cuda.synchronize()
clock0 = time.perf_counter()
for module in generator_network.modules():
    if hasattr(module, 'switch_to_deploy'):
        module.switch_to_deploy()

for i in range(300):
    prediction = generator_network(grid_sub, mask_depth, mask_depth, pre_warp_color, sample_time, pre_warp_color, sample_time)  # b c h w=1,18,h,w  1,6,h,w#generator要改channel
    # outputs = generator_network(input,noise,warp,HRwarp,warp_depth,mask)
      # Synchronize CUDA context
    # inference_time = clock1 - clock0
    # print(f"Inference time: ", inference_time)
    # sum = sum+inference_time
torch.cuda.synchronize()
clock1 = time.perf_counter()
print(f"all time: ",(clock1 - clock0) / 300)

# import time
# pre_radiance = torch.randn((2,3,192,256)).cuda()
# predepth = torch.randn((2,1,192,256)).cuda()
# mv = torch.randn((2,2,192,256)).cuda()
# prediction_lr = torch.randn((1,3,192,256)).cuda()
# depth =  torch.randn((1,1,192,256)).cuda()
# # Create CUDA events for timing
# # start_event = torch.cuda.Event(enable_timing=True)
# # end_event = torch.cuda.Event(enable_timing=True)
# # Run inference with timing
# sum = 0
# _ = nsrr_network(pre_radiance, predepth, mv, prediction_lr,depth)
# for i in range(300):
#     torch.cuda.synchronize()
#     clock0 = time.perf_counter()
#
#     _ = nsrr_network(pre_radiance, predepth, mv, prediction_lr,depth)
#      # Synchronize CUDA context
#     torch.cuda.synchronize()
#     clock1 = time.perf_counter()
#
#     inference_time = clock1 - clock0
#     sum = sum+inference_time
# alltime = sum / 300
# print(f"------------------------Inference time: " , alltime)