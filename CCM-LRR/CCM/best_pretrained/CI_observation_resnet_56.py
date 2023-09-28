import numpy as np
import torch
import os
from visdom import Visdom

use_visdom = True

if use_visdom:
    vis = Visdom(env='CI_observation_original_resnet_56')


def load_data(filename, filepath):
    x = np.load(filepath + filename, mmap_mode='r')
    return x

for i in range(55):
    filepath = "train_from_raw_resnet_56_cifar_with_ccm_test2_rl_0.01_lambda_0.01_the_best/CI_resnet_56"
    filename = "/ci_conv{}.npy".format(i+1)

    ci_conv = load_data(filename, filepath)
    print(ci_conv)
    if use_visdom:
        vis.bar(ci_conv, win="CI_for_layer{}".format(i + 1), opts=dict(title="CI_for_layer{}".format(i + 1)))


# if use_visdom:
#     vis = Visdom(env='CI_observation_contrast_1')
#
# for i in range(55):
#     filepath1 = "./CI_resnet_56"
#     filepath2 = "./CCM/result/train_from_raw_resnet_56_cifar/CI_resnet_56"
#     filepath3 = "./CCM/result/train_from_raw_resnet_56_cifar_with_ccm_test7/CI_resnet_56"
#     filename = "/ci_conv{}.npy".format(i+1)
#
#     ci_conv1 = load_data(filename, filepath1)
#     ci_conv2 = load_data(filename, filepath2)
#     ci_conv3 = load_data(filename, filepath3)
#     if use_visdom:
#         vis.bar(ci_conv1, win="CI_for_layer{}_original".format(i + 1), opts=dict(title="CI_for_layer{}_original".format(i + 1)))
#         vis.bar(ci_conv2, win="CI_for_layer{}_from_raw_resnet_56".format(i + 1), opts=dict(title="CI_for_layer{}_from_raw_resnet_56".format(i + 1)))
#         vis.bar(ci_conv3, win="CI_for_layer{}_from_raw_resnet_56_with_ccm".format(i + 1), opts=dict(title="CI_for_layer{}_from_raw_resnet_56_with_ccm".format(i + 1)))
#
