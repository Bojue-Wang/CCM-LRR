import numpy as np
import torch
import os
from visdom import Visdom

use_visdom = True

if use_visdom:
    vis = Visdom(env='CI_observation_original_VGG_16_bn')


def load_data(filename, filepath):
    x = np.load(filepath + filename, mmap_mode='r')
    return x

for i in range(12):
    filepath = "train_from_raw_vgg_16_bn_cifar_with_ccm_rl_0.01_lambda_0.01_the_best/CI_vgg_16_bn"
    filename = "/ci_conv{}.npy".format(i+1)

    ci_conv = load_data(filename, filepath)
    print(ci_conv)
    if use_visdom:
        vis.bar(ci_conv, win="CI_for_layer{}".format(i + 1), opts=dict(title="CI_for_layer{}".format(i + 1)))


