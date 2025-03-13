import torch
from math import sqrt

from sobel_warp import EvalEdgeFilter
from sobel_cpp import SobelCPPWrapper

class BaseEdgeFilter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        fx  = torch.zeros((3, 3), dtype=torch.float32)
        self.fx = torch.nn.Parameter(fx.unsqueeze(0).unsqueeze(0), 
                                     requires_grad = True)
        self.init_parameters()
    
    def init_parameters(self):
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.fx, a=sqrt(3))
        
    def forward(self, x):
        raise NotImplementedError

    def computeL1Loss(self, pred, gt):
        return torch.nn.functional.l1_loss(pred, gt)


class TorchEdgeFilter(BaseEdgeFilter):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        input = x.unsqueeze(0).unsqueeze(0)
        fy = torch.transpose(self.fx, -1, -2)
        Ix = torch.nn.functional.conv2d(input, self.fx, bias=None, padding=1)
        Iy = torch.nn.functional.conv2d(input, fy, bias=None, padding=1)
        Iedge = Ix*Ix + Iy*Iy
        return Iedge.squeeze(0).squeeze(0)

class WarpEdgeFilter(BaseEdgeFilter):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        Iedge = EvalEdgeFilter.apply(x, self.fx.squeeze(0).squeeze(0))
        return Iedge

class CppEdgeFilter(BaseEdgeFilter):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        Iedge = SobelCPPWrapper.apply(x, self.fx.squeeze(0).squeeze(0))
        return Iedge


