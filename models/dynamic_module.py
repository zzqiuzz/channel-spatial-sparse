import torch
import torch.nn as nn
import numpy as np
import utils.globalvar as gvar
class Topk(torch.autograd.Function):
    def forward(self, input):
        removed_ratio = gvar.get_value('r')
        if removed_ratio == 0:
            return input
        else:
            output = input.clone()
            temp = output.abs_()
            b, l = temp.size()
            removed_num = int(np.round(l*removed_ratio))#remove removed_num elements.
            removed_index = temp.argsort()[:,0:removed_num] # col index
            row_index = np.tile(np.arange(b),(removed_num,1)).T
            output[row_index,removed_index] = 0 
            self.save_for_backward(output)
            return output

    def backward(self, grad_output):
        output  = grad_output.clone()
        if gvar.get_value('r') != 0:
            input, = self.saved_tensors
            output[input.eq(0)] = 0
        return output
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
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
        y = self.fc(y)
        y = Topk()(y)#select topk in channel wise
        y = y.view(b, self.outchannel, 1, 1)
        return y
class DynamicSpatialConvModule(nn.Module):
    def __init__(self,planes,stride):
        super(DynamicSpatialConvModule,self).__init__()
        self.spt = nn.Sequential(
            conv3x3(2,1,stride),# max pooling + avg pooling
            nn.Sigmoid()
        ) 
        self.outchannel = planes 

    def forward(self,x):
        y_max,_ = x.max(1,keepdim=True)#b x 1 x h x w
        y_avg = x.mean(1,keepdim=True)#b x 1 x h x w
        y = torch.cat((y_max,y_avg),1)#b x 2 x h x w
        y = self.spt(y)#b x 1 x h x w 
        b,_,h,w = y.size()
        y = Topk()(y.view(b,-1))
        y = y.view(b,1,h,w)
        y = y.expand(b,self.outchannel,-1,-1) 
        return y

class DynamicSpatialFcModule(nn.Module):# spatial fc layer
    def __init__(self,spatial,planes,reduction,downsample=None):
        super(DynamicSpatialFcModule,self).__init__()
        in_d = spatial
        out_d = spatial
        if downsample:
            in_d = in_d * 2            
        spatial_in = in_d * in_d
        spatial_out = out_d * out_d
        self.fc = nn.Sequential(
            nn.Linear(spatial_in, spatial_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_in // reduction, spatial_out, bias=False),
            nn.Sigmoid()
        )
        self.out_d = out_d
        self.outchannel = planes
    def forward(self, x):
        b, c, h, w = x.size()
        y = x.mean(1,keepdim=True).view(b,-1) # out: b x hw
        y = self.fc(y)
        y = Topk()(y)#select K point in spatial wise
        y = y.view(b,1,self.out_d,self.out_d)
        y = y.expand(b,self.outchannel,-1,-1)

        return y

class DynamicBlock(nn.Module):
    def __init__(self,inplanes,planes,spatial,reduction,stride=1,downsample=None):
        super(DynamicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)        
        self.ispatial = gvar.get_value('is_spatial')
        self.dynamic_channel = DynamicChannelModule(inplanes,planes,reduction) 
        #self.dynamic_spatial = DynamicSpatialFcModule(spatial,planes,reduction,downsample) 
        if self.ispatial:
            self.dynamic_spatial = DynamicSpatialConvModule(planes,stride) 
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        channel_predictor = self.dynamic_channel(x) 
        channel_predictor.expand_as(out)
        if self.ispatial:
            spatial_predictor = self.dynamic_spatial(x)
            return channel_predictor * out * spatial_predictor #sparse in both channel and spatial 
        else:
            return channel_predictor * out

class DynamicResidualBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,spatial, stride=1, downsample=None, reduction=16):
        super(DynamicResidualBasicBlock, self).__init__()
        self.dynamicblock1 = DynamicBlock(inplanes,planes,spatial,reduction,stride,downsample)
        self.relu = nn.ReLU(inplace=True)
        self.dynamicblock2 = DynamicBlock(planes,planes,spatial,reduction)
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




