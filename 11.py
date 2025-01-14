import torch
from torch.utils.data import DataLoader
from dataset import REDS, REDSFovea
import torch.nn.functional as F
def warpFuntion(preResult: torch.Tensor, motion: torch.Tensor, curInput: torch.Tensor,
         device: torch.device = torch.device("cuda:0")):
    # preResult = preResult.unsqueeze(0)  # b c h w
    print('----------------warp-------------------')
    device = preResult.device
    motion = motion.to(device)  # b c h w
    curInput = curInput.to(device)  # b c h w
    print('motion:', motion.shape)
    print('curInput:', curInput.shape)
    print('preResult:', preResult.shape)
    b, c, h, w = preResult.shape
    x = torch.linspace(-1, 1, steps=w)
    y = torch.linspace(-1, 1, steps=h)
    grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack([grid_x, grid_y], dim=0).to(device)

    mx, my = torch.split(motion, 1, dim=1)  # channel

    mx_ = mx * 2
    my_ = my * 2

    mv_ = torch.cat([mx_, my_], dim=1).to(device)

    gridMV = (grid - mv_).permute(0, 2, 3, 1)  # .to
    warped = F.grid_sample(preResult, gridMV, align_corners=True)  # .to
    oox, ooy = torch.split((gridMV < -1) | (gridMV > 1), 1, dim=3)
    oo = (oox | ooy).permute(0, 3, 1, 2)  # .to
    print('warped:', warped.shape)
    output = torch.where(oo, curInput, warped)
    # output = output.squeeze(0)
    print('output:', output.shape)
    # ndarry = output.permute(1, 2, 0).numpy()
    # cv2.imwrite('./warped.exr', ndarry[:, :, ::-1])  # h w c,所以cv处理必须hwc
    print('----------------warp over-------------------')
    return output

validation_dataloader=DataLoader(REDSFovea(path='./Bistro/val/val_sharp'),
                                     batch_size=1, shuffle=False,num_workers=0)
generator_network = torch.load("./results/20230424-abs/models/generator_network_model_30.pt")
prevResult=None
device='cuda:0'
for index_sequence, batch in enumerate(validation_dataloader):  # 遍历每个序列

    input, label = batch  # input：img albedo normal depth mv,label:img
    print('input:', input.shape)

    motion = input[:, 12:, :, :]
    input = input[:, :12, :, :]
    print('input:', input.shape)


    if index_sequence == 0:  # warpedPrev取第一帧输入
        warpedPrev = input[:, :3, :, :]
    else:
        prevResult = prevResult.reshape((1,3,192,256))
        print('index_sequence:', index_sequence)
        print('prevResult before:', prevResult.shape)
        print('motion:', motion.shape)
        print('label:', label.shape)
        warpedPrev = warpFuntion(preResult=prevResult, motion=motion, curInput=label)
        print('warpedPrev:', warpedPrev.shape)

    masked_input = input[:, :3, :, :]  #

    input = input.to(device)  # b c h w
    masked_input = masked_input.to(device)  # b c h w
    label = label.to(device)  # b c h w
    warpedPrev = warpedPrev.to(device)  # b c h w
    prediction, pred, pred_p, pred2, pred3, halflerp, lerpResult_spatial, current, histroy, warp = generator_network(
        input.detach(), masked_input.detach(),
        warpedPrev.detach())  # b c h w=1,18,h,w  1,6,h,w#generator要改channel
    prevResult = prediction.squeeze().detach()
    print('prevResult after:', prevResult.shape)