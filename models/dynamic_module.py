import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class DynamicSpatialModule(nn.Module):
    def __init__(self,spatial_in,spatial_out,outchannel,reduction):
        super(DynamicSpatialModule,self).__init__()
        self.spatialin = spatial_in
        self.spatialout = spatial_out
        self.fc = nn.Sequential(
            nn.Linear(spatialin, spatialin // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(spatialin // reduction, spatialout, bias=False),
            nn.Sigmoid()
        )
        self.outchannel = outchannel 

    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(1,True).view(b,-1) # out: b x hw
        y = self.fc(y).view(b,self.spatialout)
        #return x * y.expand_as(x)
        # select top K largest values to output 
        # status: d = 1  select all channels
        #
        #shape = torch.Tensor(b,self.outchannel,h,w)
        
        #return y.expand_as(shape)
        return y


class DynamicChannelModule(nn.Module):
    def __init__(self,inchannel,outchannel,reduction):
        super(DynamicChannelModule,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, outchannel, bias=False),
            nn.Sigmoid()
        )
        self.outchannel = outchannel
        

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.outchannel, 1, 1)
        #return x * y.expand_as(x)
        # select top K largest values to output 
        # status: d = 1  select all channels
        #
        #shape = torch.Tensor(b,self.outchannel,h,w)
        
        #return y.expand_as(shape)
        return y


class DynamicBlock(nn.Module):
    def __init__(self,inplanes,planes,reduction,stride=1):
        super(DynamicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)        
        self.dynamic_channel = DynamicChannelModule(inplanes,planes,reduction) 
        self.dynamic_spatial = DynamicSpatialModule(spatialin,spatialout,planes,reduction) 
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        
        channel_predictor = self.dynamic_channel(x) 
        channel_predictor.expand_as(out)
        spatial_predictor = self.dynamic_spatial(x)
        spatial_predictor.
        return channel_predictor * out 

class DynamicResidualBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(DynamicResidualBasicBlock, self).__init__()
        self.dynamicblock1 = DynamicBlock(inplanes,planes,reduction,stride)
        self.relu = nn.ReLU(inplace=True)
        self.dynamicblock2 = DynamicBlock(planes,planes,reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.dynamicblock1(x)
        out = self.relu(out)
        out = self.dynamicblock2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




