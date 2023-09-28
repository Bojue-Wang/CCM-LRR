import os
import numpy as np
import time, datetime
import torch
import argparse
import math
import shutil
from collections import OrderedDict
from thop import profile

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from models.resnet_imagenet import resnet_50
from models.resnet_cifar10 import resnet_56, resnet_110

from data import imagenet
import utils
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model
model = resnet_110(sparsity=[0.]*100).to(device)

# calculate model size
input_image_size = 32
input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
flops, params = profile(model, inputs=(input_image,))
print('Params: %.2f' % (params))
print('Flops: %.2f' % (flops))
print(model)
