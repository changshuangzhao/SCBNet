import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv2d(in_channel, out_channel, kernel_size, stride, pad, dilation):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation)


def convbn_3d(in_channel, out_channel, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=pad, stride=stride),
                         nn.BatchNorm3d(out_channel))


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, pad=1, dilation=1):
        super(ResBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Sequential(conv2d(in_channel, out_channel, 3, stride, pad, dilation),
                                  nn.BatchNorm2d(in_channel),
                                  nn.ReLU(True),
                                  conv2d(out_channel, out_channel, 3, 1, pad, dilation),
                                  nn.BatchNorm2d(in_channel))
        self.downsample = downsample

    def forward(self, x):
        out = self.bn(x)
        if self.downsample is not None:
            x = self.downsample(out)
        out = self.conv(out)

        out = x + out
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, stride=1, pad=1, dilation=1):
        super(DenseBlock, self).__init__()
        self.conv = nn.Sequential(nn.BatchNorm2d(in_planes),
                                  nn.ReLU(True),
                                  nn.Conv2d(in_planes, growth_rate, 3, stride, dilation if dilation > 1 else pad, dilation))

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], 1)


class SkipDenseBlock(nn.Module):
    def __init__(self, in_plane, dense_astrous_list, growth_rate):
        super(SkipDenseBlock, self).__init__()
        self.residual_astrous_blocks = nn.ModuleList()
        for di in dense_astrous_list:
            self.residual_astrous_blocks.append(DenseBlock(in_plane, growth_rate, stride=1, pad=1, dilation=di))
            in_plane = in_plane + growth_rate
        self.conv2d_out = nn.Sequential(nn.BatchNorm2d(in_plane),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_plane, in_plane // 2, 1, 1, 0),
                                        nn.BatchNorm2d(in_plane // 2),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_plane // 2, 32, 3, 1, 1, bias=False))

    def forward(self, x):
        for astrous_block in self.residual_astrous_blocks:
            x = astrous_block(x)
        output = self.conv2d_out(x)
        return output


class SkipResBlock(nn.Module):
    def __init__(self, in_channel, res_astrous_list):
        super(SkipResBlock, self).__init__()
        self.conv2d_in = nn.Conv2d(in_channel, 32, 3, 1, 1)
        self.residual_astrous_blocks = nn.ModuleList()
        for di in res_astrous_list:
            self.residual_astrous_blocks.append(ResBlock(32, 32, stride=1, downsample=None, pad=1, dilation=di))
        self.conv2d_out = nn.Sequential(nn.BatchNorm2d(32),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, low_disparity, corresponding_rgb):
        twice_disparity = F.interpolate(low_disparity, size=corresponding_rgb.size()[-2:], mode='bilinear', align_corners=False)
        if corresponding_rgb.size()[-1] / low_disparity.size()[-1] >= 1.5:
            twice_disparity *= 2
        output = self.conv2d_in(torch.cat([twice_disparity, corresponding_rgb], dim=1))
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)

        return nn.ReLU(True)(twice_disparity + self.conv2d_out(output))


class HourGlass(nn.Module):
    def __init__(self, inplanes):
        super(HourGlass, self).__init__()
        self.conv0 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(True),
                                   convbn_3d(inplanes, inplanes, 3, 1, 1))

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(True),
                                   convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1))

        self.conv2 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(True),
                                   convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(True))

        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                     nn.BatchNorm3d(inplanes * 2))

        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                     nn.BatchNorm3d(inplanes))

        self.conv3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(True),
                                   nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        out = self.conv2(conv1)
        out = self.relu(self.deconv1(out) + conv1)
        out = self.relu(self.deconv2(out) + conv0)
        out = self.conv3(out)
        return out


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.disp = torch.FloatTensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1, True)
        return out


class CorrelationLayer1D(nn.Module):
    def __init__(self, max_disp=20, stride_2=1):
        super(CorrelationLayer1D, self).__init__()
        self.max_displacement = max_disp
        self.stride_2 = stride_2

    def forward(self, x_1, x_2):
        x_1 = x_1
        x_2 = F.pad(x_2, (self.max_displacement, self.max_displacement, 0, 0))
        return torch.cat([torch.sum(x_1 * x_2[:, :, :, _y:_y + x_1.size(3)], 1, keepdim=True) for _y in
                          range(0, self.max_displacement * 2 + 1, self.stride_2)], 1)
