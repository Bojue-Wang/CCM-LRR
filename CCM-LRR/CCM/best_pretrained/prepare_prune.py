import numpy as np
import math
from models.resnet_cifar10 import resnet_56, resnet_110, resnet_56_spa
import torch
from thop import profile


def prepare_prune(ci_dir, layer, threshold):
    prefix = ci_dir + '/ci_conv'
    subfix = ".npy"
    channel_reserved_by_CI = []
    kernel_count_for_each_layer = []
    for i in range(layer):
        ci = np.load(prefix + str(i + 1) + subfix)
        ci_sum = np.sum(ci)
        ci_sorted = np.argsort(ci)
        select_index = []
        ci_temp = 0
        for index in range(len(ci_sorted)):
            ci_temp += ci[ci_sorted[len(ci_sorted) - 1 - index]]
            select_index.append(ci_sorted[len(ci_sorted) - 1 - index])
            if ci_temp > threshold * ci_sum:
                break
            else:
                continue
        select_index = np.array(select_index)
        select_index.sort()
        channel_reserved_by_CI.append(select_index)
        kernel_count_for_each_layer.append(len(select_index))
    return channel_reserved_by_CI, kernel_count_for_each_layer


# ci_dir_56 = 'train_from_raw_resnet_56_cifar_with_ccm_test2_rl_0.01_lambda_0.01_the_best/CI_resnet_56'
# ci_dir_110 = 'train_from_raw_resnet_110_cifar_with_ccm_rl_0.01_lambda_0.004_the_best/CI_resnet_110'
# ci_dir_16 = 'train_from_raw_vgg_16_bn_cifar_with_ccm_rl_0.01_lambda_0.01_the_best/CI_vgg_16_bn'

# channel_reserved_by_CI_56, kernels_56 = prepare_prune(ci_dir_56, 55, 0.7)
# channel_reserved_by_CI_56_1, kernels_56_1 = prepare_prune(ci_dir_56, 55, 1)
# print(channel_reserved_by_CI_56)
# print(kernels_56)
# print(len(kernels_56))
# kernels_percentage = []
# for i in range(len(kernels_56)):
#     kernels_percentage.append(kernels_56[i]/kernels_56_1[i])
# print(kernels_percentage)

# channel_reserved_by_CI_110, kernels_110 = prepare_prune(ci_dir_110, 109, 0.7)
# print(channel_reserved_by_CI_110)
# print(kernels_110)
# print(len(kernels_110))

# channel_reserved_by_CI_16, kernels_16 = prepare_prune(ci_dir_16, 12, 0.7)
# print(channel_reserved_by_CI_16)
# print(kernels_16)
# print(len(kernels_16))

