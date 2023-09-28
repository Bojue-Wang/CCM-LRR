import torch.nn as nn
from collections import OrderedDict
import numpy as np

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]

class VGG(nn.Module):
    def __init__(self, sparsity, cfg=None, num_classes=10):
        super(VGG, self).__init__()

        if cfg is None:
            cfg = defaultcfg
        self.relucfg = relucfg

        self.sparsity = sparsity[:]
        self.sparsity.append(0.0)
        print(self.sparsity)

        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt=0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                x = int(x * (1-self.sparsity[cnt]))
                print(x)

                cnt+=1
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg_16_bn(sparsity):
    return VGG(sparsity=sparsity)



class VGG_spa(nn.Module):
    def __init__(self, sparsity, cfg=None, num_classes=10, sparsity_type="original"):
        super(VGG_spa, self).__init__()

        if cfg is None:
            cfg = defaultcfg
        self.relucfg = relucfg
        self.sparsity_type = sparsity_type

        if self.sparsity_type == "original":
            self.sparsity = sparsity[:]
            self.sparsity.append(0.0)
        elif self.sparsity_type == "ccm":
            self.sparsity = sparsity[:]
            self.sparsity.append(512)

        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

    def _make_layers(self, cfg):

        layers = nn.Sequential()
        in_channels = 3
        cnt=0

        for i, x in enumerate(cfg):
            if x == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                if self.sparsity_type == "original":
                    x = int(x * (1-self.sparsity[cnt]))
                elif self.sparsity_type == "ccm":
                    x = int(self.sparsity[cnt])

                cnt+=1
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(x))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = x

        return layers

    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg_16_bn_spa(sparsity, sparsity_type="original"):
    return VGG_spa(sparsity=sparsity, sparsity_type=sparsity_type)

#
# spa = [13, 39, 81, 80, 153, 142, 143, 275, 167, 123, 82, 37]
# vgg_test = vgg_16_bn_spa(sparsity=spa, sparsity_type='ccm')
# print(vgg_test)
#
# # sparsity = '[0.3]*7+[0.75]*5'
# #
# # import re
# # cprate_str = sparsity
# # cprate_str_list = cprate_str.split('+')
# # pat_cprate = re.compile(r'\d+\.\d*')
# # pat_num = re.compile(r'\*\d+')
# # cprate = []
# # for x in cprate_str_list:
# #     num = 1
# #     find_num = re.findall(pat_num, x)
# #     if find_num:
# #         assert len(find_num) == 1
# #         num = int(find_num[0].replace('*', ''))
# #     find_cprate = re.findall(pat_cprate, x)
# #     assert len(find_cprate) == 1
# #     cprate += [float(find_cprate[0])] * num
# # sparsity = cprate
# # vgg_16_bn_spa(sparsity, sparsity_type="original")
# #
# #
# # def prepare_prune(ci_dir, layer, threshold):
# #     prefix = ci_dir + '/ci_conv'
# #     subfix = ".npy"
# #     channel_reserved_by_CI = []
# #     kernel_count_for_each_layer = []
# #     for i in range(layer):
# #         ci = np.load(prefix + str(i + 1) + subfix)
# #         ci_sum = np.sum(ci)
# #         ci_sorted = np.argsort(ci)
# #         select_index = []
# #         ci_temp = 0
# #         for index in range(len(ci_sorted)):
# #             ci_temp += ci[ci_sorted[len(ci_sorted) - 1 - index]]
# #             select_index.append(ci_sorted[len(ci_sorted) - 1 - index])
# #             if ci_temp > threshold * ci_sum:
# #                 break
# #             else:
# #                 continue
# #         select_index = np.array(select_index)
# #         select_index.sort()
# #         channel_reserved_by_CI.append(select_index)
# #         kernel_count_for_each_layer.append(len(select_index))
# #     return channel_reserved_by_CI, kernel_count_for_each_layer
# #
# # ci_dir_16 = '../CCM/best_pretrained/train_from_raw_vgg_16_bn_cifar_with_ccm_rl_0.01_lambda_0.01_the_best/CI_vgg_16_bn'
# #
# # channel_reserved_by_CI_16, kernels_16 = prepare_prune(ci_dir_16, 12, 0.7)
# # # print(channel_reserved_by_CI_16)
# # print(kernels_16)

# sparsity = "[0.45]*7+[0.78]*5"
sparsity = "[0.0]*7+[0.0]*5"
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

net = VGG(sparsity)
print(net)