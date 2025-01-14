import torch  
import torch.nn as nn  

class L1Loss(nn.Module):  
    def __init__(self, reduction='mean'):  
        super(L1Loss, self).__init__()  
        self.reduction = reduction  

    def forward(self, input, target):  
        # 计算绝对误差  
        loss = (input - target).abs()
        
        # 根据 reduction 类型应用不同的处理  
        if self.reduction == 'mean':  
            return loss.mean()  
        elif self.reduction == 'sum':  
            return loss.sum()  
        else:  # 'none'  
            return loss  
