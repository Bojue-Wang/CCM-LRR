
import torch.nn as nn
import torch.nn.functional as F

def adapt_channel(sparsity, num_layers):

    if num_layers==56:
        stage_repeat = [9, 9, 9]
        stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9
    elif num_layers==110:
        stage_repeat = [18, 18, 18]
        stage_out_channel = [16] + [16] * 18 + [32] * 18 + [64] * 18

    stage_oup_cprate = []
    stage_oup_cprate += [sparsity[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [sparsity[i+1]] * stage_repeat[i]
    stage_oup_cprate +=[0.] * stage_repeat[-1]
    mid_cprate = sparsity[len(stage_repeat):]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i] * (1-mid_cprate[i-1]))]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, midplanes, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            #self.shortcut = LambdaLayer(
            #    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),"constant", 0))

            '''self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                #nn.BatchNorm2d(planes),
            )#'''

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #print(self.stride, self.inplanes, self.planes, out.size(), self.shortcut(x).size())
        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, sparsity, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        self.overall_channel, self.mid_channel = adapt_channel(sparsity, num_layers)
        print(self.overall_channel)
        print(self.mid_channel)

        self.layer_num = 0
        self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.overall_channel[self.layer_num])
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()
        self.layer_num += 1

        #self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.num_layer == 56:
            self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
        else:
            self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)


    def _make_layer(self, block, blocks_num, stride):
        layers = []
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                 self.overall_channel[self.layer_num], stride))
        self.layer_num += 1

        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num]))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        return x


def resnet_56(sparsity):
    return ResNet(BasicBlock, 56, sparsity=sparsity)

def resnet_110(sparsity):
    return ResNet(BasicBlock, 110, sparsity=sparsity)


class ResNet_spa(nn.Module):
    def __init__(self, block, num_layers, sparsity, num_classes=10, sparsity_type="original"):
        super(ResNet_spa, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        if sparsity_type == "original":
            self.overall_channel, self.mid_channel = adapt_channel(sparsity, num_layers)
        elif sparsity_type == "ccm":
            self.overall_channel = []
            self.mid_channel = []
            for i in range(len(sparsity)):
                if i % 2 == 0:
                    self.overall_channel.append(sparsity[i])
                else:
                    self.mid_channel.append(sparsity[i])

        self.layer_num = 0
        self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.overall_channel[self.layer_num])
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()
        self.layer_num += 1

        # self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(block, blocks_num=n, stride=1)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if sparsity_type == "original":
            if self.num_layer == 56:
                self.fc = nn.Linear(64 * BasicBlock.expansion, num_classes)
            else:
                self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)
        elif sparsity_type == "ccm":
            if self.num_layer == 56:
                self.fc = nn.Linear(sparsity[-1] * BasicBlock.expansion, num_classes)
            else:
                self.linear = nn.Linear(sparsity[-1] * BasicBlock.expansion, num_classes)


    def _make_layer(self, block, blocks_num, stride):
        layers = []
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                            self.overall_channel[self.layer_num], stride))
        self.layer_num += 1

        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                self.overall_channel[self.layer_num]))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        return x


def resnet_56_spa(sparsity, sparsity_type='ccm'):
    return ResNet_spa(BasicBlock, 56, sparsity=sparsity, sparsity_type=sparsity_type)

def resnet_110_spa(sparsity, sparsity_type='ccm'):
    return ResNet_spa(BasicBlock, 110, sparsity=sparsity, sparsity_type=sparsity_type)

# sparsity =  "[0.]+[0.4]*2+[0.5]*9+[0.6]*9+[0.7]*9"
sparsity =  "[0.]+[0.0]*2+[0.0]*9+[0.0]*9+[0.0]*9"
if sparsity:
    import re
    cprate_str = sparsity
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    sparsity = cprate

print(sparsity)
net = ResNet(BasicBlock, 56, sparsity=sparsity)
print(net)