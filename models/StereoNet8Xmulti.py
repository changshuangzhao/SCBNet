from .submodel import *


class StereoNet(nn.Module):
    def __init__(self, maxdisp=192):
        super().__init__()
        self.maxdisp = maxdisp // 4

        self.inplanes = 32
        self.conv_0 = nn.Conv2d(3, 32, 5, 2, 2)

        self.layer1 = self._make_layer(ResBlock, 32, 3)
        self.layer2 = self._make_layer(ResBlock, 32, 4, stride=2)
        self.layer1_1x1 = nn.Sequential(nn.BatchNorm2d(32),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 16, 3, 1, 1, bias=False))

        dense_astrous_list = [1, 2, 4, 8, 16, 1]
        self.skipdenseblock = SkipDenseBlock(32, dense_astrous_list, 16)

        self.hourglass = HourGlass(32)

        self.dispparityregression = DisparityRegression(self.maxdisp)
        self.corr1D_layer = CorrelationLayer1D(20)
        self.corr1D_image = CorrelationLayer1D(10)

        res_astrous_list = [1, 2, 4, 8, 1, 1]
        self.skipresblock1 = SkipResBlock(42, res_astrous_list)
        self.skipresblock2 = SkipResBlock(22, res_astrous_list)

    def forward(self, left, right):
        refimg_feature = self.conv_0(left)
        refimg_layer1 = self.layer1(refimg_feature)
        refimg_layer2 = self.layer2(refimg_layer1)
        refimg = self.skipdenseblock(refimg_layer2)
        layer1_1x1_refer = self.layer1_1x1(refimg_layer1)

        targetimg_feature = self.conv_0(right)
        targetimg_layer1 = self.layer1(targetimg_feature)
        targetimg_layer2 = self.layer2(targetimg_layer1)
        targetimg = self.skipdenseblock(targetimg_layer2)
        layer1_1x1_target = self.layer1_1x1(targetimg_layer1)

        cost = torch.FloatTensor(refimg.size()[0], refimg.size()[1], self.maxdisp, refimg.size()[2], refimg.size()[3]).zero_().cuda()
        for i in range(self.maxdisp):
            if i > 0:
                cost[:, :, i, :, i:] = refimg[:, :, :, i:] - targetimg[:, :, :, :-i]
            else:
                cost[:, :, i, :, :] = refimg - targetimg
        cost = cost.contiguous()

        cost = self.hourglass(cost)
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost, dim=1)
        pred = self.dispparityregression(pred)

        corr1D_layer = self.corr1D_layer(layer1_1x1_refer, layer1_1x1_target)
        corr1D_image = self.corr1D_image(left, right)

        pred_pyramid_list = []
        pred_pyramid_list.append(pred)
        pred_pyramid_list.append(self.skipresblock1(pred_pyramid_list[0], corr1D_layer))
        pred_pyramid_list.append(self.skipresblock2(pred_pyramid_list[1], corr1D_image))

        length_all = len(pred_pyramid_list)  # 3

        for i in range(length_all):
            pred_pyramid_list[i] = pred_pyramid_list[i] * (left.size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = torch.squeeze(F.interpolate(pred_pyramid_list[i], size=left.size()[-2:], mode='bilinear', align_corners=False), dim=1)

        return pred_pyramid_list  # 4

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(self.inplanes))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)






    


