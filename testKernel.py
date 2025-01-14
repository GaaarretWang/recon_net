import torch

import torch.nn.functional as F
def _convert_dict(core, batch_size, N, color, height, width):
    """
    make sure the core to be a dict, generally, only one kind of kernel size is suitable for the func.
    :param core: shape: batch_size*(N*K*K)*height*width (1,18*3*3,1017,1920)
    :return: core_out, a dict
    """

    # revise
    # 1,25,720,1280
    # core=core.unsqueeze(0);
    # 1,1,25,720,1280
    # core = core.unsqueeze(3);
    # 1,1,25,1,720,1280
    # core = torch.cat((core, core, core), dim=3)
    # 1,1,25,3,720,1280

    core = core.view(batch_size, N, -1, color, height, width)  # 1 2 25 3 192 256
    # -1 表示 PyTorch 自动计算缺失的维度大小

    core = F.softmax(core, dim=2)  # 对5*5的卷积核的权重做softmax，将其归一化在[0,1]
    return core
output = torch.randn(1,3,192,256)
print(output.shape)

kernel_weight = torch.randn(1,75,192,256)#1 25*3*2 192 256
print(kernel_weight.shape)

kernel=5

batch_size, N, height, width = output.size()
print('batch_size:',batch_size)
print('N:',N)
print('height:',height)
print('width:',width)
print('output.size():',output.size())
color = 3
N=1#N=2
frames = output.view(batch_size, N, color, height, width)#  1 1 3 192 256
print('frames:' , frames.shape)

core = _convert_dict(kernel_weight, batch_size, N, color, height, width)#（1,1,25,3,192,256）
print('core:' , core.shape)

img_stack = []
pred_img = []

if not img_stack:
    frame_pad = F.pad(frames, [kernel // 2, kernel // 2, kernel // 2, kernel // 2])  # 在输入图像周围加一圈宽2的像素
    # frame_pad:1,2,3,196,260
    print('frame_pad:' , frame_pad.shape)
    for i in range(kernel):
        for j in range(kernel):
            img_stack.append(frame_pad[..., i:i + height, j:j + width])#1 2 3 192 256

    img_stack = torch.stack(img_stack, dim=2)#1 1 25 3 192 256
    print('len:',len(img_stack))
    print(img_stack.shape)

#core:1 1 25 3 192 256
#img_stack:1 1 25 3 192 256
m = core.mul(img_stack)
print('m' , m.shape)#1 1 25 3 192 256

pred_img.append(torch.sum(m, dim=2, keepdim=False)) #sum:1 2 3 192 256
print('pred_img len:',len(pred_img))#1

pred_img = torch.stack(pred_img, dim=0)
print('pred_img:',pred_img.shape)#1 1 1 3 192 256

pred_img_i = torch.mean(pred_img, dim=0, keepdim=False)
print('pred_img_i:',pred_img_i.shape)#1 1 3 192 256  batch上融合？

pred_img = torch.mean(pred_img_i, dim=1, keepdim=False)
print('pred_img:',pred_img.shape)#1 3 192 256
