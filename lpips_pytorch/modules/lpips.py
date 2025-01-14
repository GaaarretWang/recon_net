import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor, all_fovea_masks: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]

        downsample_layers = nn.ModuleList([  
            nn.Identity(),  # 原图，1024x1024  
            nn.AvgPool2d(kernel_size=2),  # 2倍降采样，512x512  
            nn.AvgPool2d(kernel_size=4),  # 4倍降采样，256x256  
            nn.AvgPool2d(kernel_size=8),  # 8倍降采样，128x128  
            nn.AvgPool2d(kernel_size=16),  # 16倍降采样，64x64  
        ])  
        downsampled_images = []  
        for layer in downsample_layers:  
            downsampled_image = layer(all_fovea_masks)  
            downsampled_images.append(downsampled_image)  

        diff_weighted = [d * mask for d, mask in zip(diff, downsampled_images)]  
        res = [l(d).mean((2, 3), True) for d, l in zip(diff_weighted, self.lin)]
        # res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
