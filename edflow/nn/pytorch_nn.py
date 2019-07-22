import torch
import torch.nn as nn

class CoordConv2d(nn.Module):
    """
    CoordConv layer as described in https://arxiv.org/abs/1807.03247
    Appends relative x and y coordinates to input tensor and perfroms convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode="zeros"):
        super(CoordConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels+2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
 
    def forward(self, x):
        #append coordinate channels to x (make sure those are floats)
        gy, gx = torch.meshgrid(torch.arange(0.0, x.size(2)), torch.arange(0.0, x.size(3)))
        
        #normalize y and x to [0,1]
        gy = gy.view(1, 1, x.size(2), x.size(3))/float(x.size(2))
        gx = gx.view(1, 1, x.size(2), x.size(3))/float(x.size(3))
        gy = gy.expand(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        gx = gx.expand(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        
        x = torch.cat([x, gy, gx], dim=1)
        
        return self.conv(x)


class CoordConvTranspose2d(nn.Module):
    """
    CoordConv layer as described in https://arxiv.org/abs/1807.03247
    Appends relative x and y coordinates to input tensor and perfroms transposed convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode="zeros"):
        super(CoordConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels+2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, padding_mode=padding_mode)
    
    def forward(self, x):
        #append coordinate channels to x (make sure those are floats
        gy, gx = torch.meshgrid(torch.arange(0.0, x.size(2)), torch.arange(0.0, x.size(3)))
        
        #normalize y and x to [0,1]
        gy = gy.view(1, 1, x.size(2), x.size(3))/float(x.size(2))
        gx = gx.view(1, 1, x.size(2), x.size(3))/float(x.size(3))
        gy = gy.expand(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        gx = gx.expand(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        
        x = torch.cat([x, gy, gx], dim=1)
        
        return self.conv(x)
