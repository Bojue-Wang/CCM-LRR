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
from data import cifar10
import utils
from torch.cuda.amp import autocast, GradScaler
from CCM_model import loss_per_layer, channel_correlation_matrix_GPU, save_coco_M
import sys
import copy
import torch.utils.data.distributed
import torchvision
from torchvision import datasets, transforms

sys.path.append("../..")

parser = argparse.ArgumentParser("ImageNet training resnet_50")
parser.add_argument('--data_dir', type=str, default='data', help='path to dataset')
parser.add_argument('--arch', type=str, default='resnet_56',
                    choices=('vgg_16_bn', 'resnet_56', 'resnet_110', 'resnet_50'),
                    help='architecture to calculate feature maps')
parser.add_argument('--lr_type', type=str, default='cos', help='lr type')
parser.add_argument('--result_dir', type=str, default='./result/train_from_raw_resnet_50_imagenet',
                    help='results path for saving models and loggers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
parser.add_argument('--label_smooth', type=float, default=0, help='label smoothing')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--lr_decay_step', default='30,60', type=str, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
parser.add_argument('--pretrain_dir', type=str, default='', help='pretrain model path')
parser.add_argument('--ci_dir', type=str, default='', help='ci path')
parser.add_argument('--sparsity', type=str, default=None, help='sparsity of each conv layer')
parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--visdom', type=bool, default=False, help='use visdom or not')
args = parser.parse_args()

args.data_dir = '../../data'
args.arch = 'resnet_50'
args.lr_type = 'cos'
args.result_dir = './result/train_from_raw_resnet_50_imagenet'
args.batch_size = 256
args.epochs = 180
args.label_smooth = 0.1
args.learning_rate = 0.01
args.lr_decay_step = '30,60'
args.momentum = 0.99
args.weight_decay = 0.0001
args.pretrain_dir = ''
args.ci_dir = ''
args.sparsity = None
args.gpu = '3'
args.visdom = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.visdom:
    from visdom import Visdom
    vis = Visdom(env='train_from_raw_resnet_50')
    vis.line([[0.0, 0.0]], [0.], win='CE_loss & CCM_loss.',
             opts=dict(title='CE_loss & CCM_loss.', legend=['CE_loss', 'CCM_loss']))
    vis.line([[0.0, 0.0]], [0.], win='Train & Test Accuracy.',
             opts=dict(title='Train & Test Accuracy.', legend=['Train acc',
                                                               'Test acc']))

CLASSES = 1000
print_freq = 128000 // args.batch_size

if not os.path.isdir(args.result_dir):
    os.makedirs(args.result_dir)

# save old training file
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
cp_file_dir = os.path.join(args.result_dir, 'cp_file/' + now)
if os.path.exists(args.result_dir + '/model_best.pth.tar'):
    if not os.path.isdir(cp_file_dir):
        os.makedirs(cp_file_dir)
    shutil.copy(args.result_dir + '/config.txt', cp_file_dir)
    shutil.copy(args.result_dir + '/logger.log', cp_file_dir)
    shutil.copy(args.result_dir + '/model_best.pth.tar', cp_file_dir)
    shutil.copy(args.result_dir + '/checkpoint.pth.tar', cp_file_dir)

utils.record_config(args)
logger = utils.get_logger(os.path.join(args.result_dir, 'logger.log'))

# use for loading pretrain model
if len(args.gpu) > 1:
    name_base = 'module.'
else:
    name_base = ''


def load_resnet_model(model, oristate_dict):
    cfg = {'resnet_50': [3, 4, 6, 3], }

    state_dict = model.state_dict()

    current_cfg = cfg[args.arch]
    last_select_index = None

    all_honey_conv_weight = []

    bn_part_name = ['.weight', '.bias', '.running_mean', '.running_var']  # ,'.num_batches_tracked']
    prefix = args.ci_dir + '/ci_conv'
    subfix = ".npy"
    cnt = 1

    conv_weight_name = 'conv1.weight'
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[name_base + conv_weight_name]
    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num:
        logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
        ci = np.load(prefix + str(cnt) + subfix)
        select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
        select_index.sort()

        for index_i, i in enumerate(select_index):
            state_dict[name_base + conv_weight_name][index_i] = \
                oristate_dict[conv_weight_name][i]
            for bn_part in bn_part_name:
                state_dict[name_base + 'bn1' + bn_part][index_i] = \
                    oristate_dict['bn1' + bn_part][i]

        last_select_index = select_index
    else:
        state_dict[name_base + conv_weight_name] = oriweight
        for bn_part in bn_part_name:
            state_dict[name_base + 'bn1' + bn_part] = oristate_dict['bn1' + bn_part]

    state_dict[name_base + 'bn1' + '.num_batches_tracked'] = oristate_dict['bn1' + '.num_batches_tracked']

    cnt += 1
    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            iter = 3
            if k == 0:
                iter += 1
            for l in range(iter):
                record_last = True
                if k == 0 and l == 2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                    bn_name = layer_name + str(k) + '.downsample.1'
                    record_last = False
                elif k == 0 and l == 3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                    bn_name = layer_name + str(k) + '.bn' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                    bn_name = layer_name + str(k) + '.bn' + str(l + 1)

                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[name_base + conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num:
                    logger.info('loading ci from: ' + prefix + str(cnt) + subfix)
                    ci = np.load(prefix + str(cnt) + subfix)
                    select_index = np.argsort(ci)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]

                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base + conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                            for bn_part in bn_part_name:
                                state_dict[name_base + bn_name + bn_part][index_i] = \
                                    oristate_dict[bn_name + bn_part][i]

                    if record_last:
                        last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]

                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]

                    if record_last:
                        last_select_index = None

                else:
                    state_dict[name_base + conv_weight_name] = oriweight
                    for bn_part in bn_part_name:
                        state_dict[name_base + bn_name + bn_part] = \
                            oristate_dict[bn_name + bn_part]
                    if record_last:
                        last_select_index = None

                state_dict[name_base + bn_name + '.num_batches_tracked'] = oristate_dict[
                    bn_name + '.num_batches_tracked']
                cnt += 1

    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[name_base + conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name_base + name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name_base + name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, epoch, step, len_iter):
    if args.lr_type == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.1 ** factor)

    elif args.lr_type == 'step_5':
        factor = epoch // 10
        if epoch >= 80:
            factor = factor + 1
        lr = args.learning_rate * (0.5 ** factor)

    elif args.lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - 5) / (args.epochs - 5)))

    elif args.lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = args.learning_rate * (decay ** (epoch // step))

    elif args.lr_type == 'fixed':
        lr = args.learning_rate
    else:
        raise NotImplementedError

    # Warmup
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_iter) / (5. * len_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if step == 0:
        logger.info('learning_rate: ' + str(lr))


def train(epoch, train_loader, model, criterion, optimizer, scaler=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    CCM_losses = utils.AverageMeter('CCM_loss', ':.4e')

    model.train()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

        ################################
        feature_map_box = []
        hook_list = []

        def feature_map_temp(model, input, output):
            feature_map_box.append(output)

        if args.arch == 'resnet_56':
            cov_layer = eval('model.relu')
            handler = cov_layer.register_forward_hook(feature_map_temp)
            hook_list.append(handler)
            # ResNet56 per block
            for k in range(3):
                block = eval('model.layer%d' % (k + 1))
                for j in range(9):
                    cov_layer = block[j].relu1
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)

                    cov_layer = block[j].relu2
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)

        elif args.arch == 'resnet_50':
            cov_layer = eval('model.maxpool')
            handler = cov_layer.register_forward_hook(feature_map_temp)
            hook_list.append(handler)
            # ResNet50 per bottleneck
            for i in range(4):
                block = eval('model.layer%d' % (i + 1))
                for j in range(model.num_blocks[i]):
                    cov_layer = block[j].relu1
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)
                    cov_layer = block[j].relu2
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)
                    cov_layer = block[j].relu3
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)
                    if j == 0:
                        cov_layer = block[j].relu3
                        handler = cov_layer.register_forward_hook(feature_map_temp)
                        hook_list.append(handler)
        #################################################

        # compute output
        logits = model(images)
        data_loss = criterion(logits, targets)
        CCM_loss = 0
        for item in feature_map_box:
            CCM_loss += loss_per_layer(item)

        loss = data_loss

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(data_loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        CCM_losses.update(CCM_loss, n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # release hooks
        for item in hook_list:
            item.remove()
        feature_map_box = []

        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg, CCM_losses.avg


def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    CCM_losses = utils.AverageMeter('CCM_loss', ':.4e')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            ################################
            feature_map_box = []
            hook_list = []

            def feature_map_temp(model, input, output):
                feature_map_box.append(output)

            if args.arch == 'resnet_56':
                cov_layer = eval('model.relu')
                handler = cov_layer.register_forward_hook(feature_map_temp)
                hook_list.append(handler)
                # ResNet56 per block
                for k in range(3):
                    block = eval('model.layer%d' % (k + 1))
                    for j in range(9):
                        cov_layer = block[j].relu1
                        handler = cov_layer.register_forward_hook(feature_map_temp)
                        hook_list.append(handler)

                        cov_layer = block[j].relu2
                        handler = cov_layer.register_forward_hook(feature_map_temp)
                        hook_list.append(handler)


            elif args.arch == 'resnet_50':
                cov_layer = eval('model.maxpool')
                handler = cov_layer.register_forward_hook(feature_map_temp)
                hook_list.append(handler)
                # ResNet50 per bottleneck
                for i in range(4):
                    block = eval('model.layer%d' % (i + 1))
                    for j in range(model.num_blocks[i]):
                        cov_layer = block[j].relu1
                        handler = cov_layer.register_forward_hook(feature_map_temp)
                        hook_list.append(handler)
                        cov_layer = block[j].relu2
                        handler = cov_layer.register_forward_hook(feature_map_temp)
                        hook_list.append(handler)
                        cov_layer = block[j].relu3
                        handler = cov_layer.register_forward_hook(feature_map_temp)
                        hook_list.append(handler)
                        if j == 0:
                            cov_layer = block[j].relu3
                            handler = cov_layer.register_forward_hook(feature_map_temp)
                            hook_list.append(handler)
            #################################################

            # compute output
            logits = model(images)
            loss = criterion(logits, target)
            CCM_loss = 0
            for item in feature_map_box:
                CCM_loss += loss_per_layer(item)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)
            CCM_losses.update(CCM_loss, n)

            # release hook
            for item in hook_list:
                item.remove()
            feature_map_box = []

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg, CCM_losses.avg


def main():
    start_t = time.time()
    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)

    # load model
    logger.info('==> Building model..')
    model = eval(args.arch)(sparsity=[0.] * 100).cuda()
    logger.info(model)

    # calculate model size
    input_image_size = 224
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))

    # load training data
    print('==> Preparing data..')
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.train_loader
    val_loader = data_tmp.test_loader

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    start_epoch = 0
    best_top1_acc = 0
    best_top5_acc = 0

    # train the model
    scaler = GradScaler()
    epoch = start_epoch
    while epoch < args.epochs:

        train_obj, train_top1_acc, train_top5_acc, train_CCM_loss = train(epoch, train_loader, model, criterion_smooth, optimizer,
                                                          scaler)
        valid_obj, valid_top1_acc, valid_top5_acc, valid_CCM_loss = validate(epoch, val_loader, model, criterion, args)

        # update visdom painting
        if args.visdom:
            vis.line([[valid_obj, valid_CCM_loss.item() * 0.1]], [epoch], win='CE_loss & CCM_loss.', update='append')

            vis.line([[train_top1_acc, valid_top1_acc.item()]], [epoch], win='Train & Test Accuracy.', update='append')

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            best_top5_acc = valid_top5_acc
            is_best = True

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'best_top5_acc': best_top5_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.result_dir)

        epoch += 1
        logger.info("=>Best accuracy Top1: {:.3f}, Top5: {:.3f}".format(best_top1_acc, best_top5_acc))

    training_time = (time.time() - start_t) / 36000
    logger.info('total training time = {} hours'.format(training_time))

    # print coco_matrix
    if args.visdom:
        model.eval()
        correct_count_in_train_set = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.cuda(), target.cuda()

                ################################
                feature_map_box = []
                hook_list = []

                def feature_map_temp(model, input, output):
                    feature_map_box.append(output)

                if args.arch == 'resnet_56':
                    cov_layer = eval('model.relu')
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)
                    # ResNet56 per block
                    for k in range(3):
                        block = eval('model.layer%d' % (k + 1))
                        for j in range(9):
                            cov_layer = block[j].relu1
                            handler = cov_layer.register_forward_hook(feature_map_temp)
                            hook_list.append(handler)

                            cov_layer = block[j].relu2
                            handler = cov_layer.register_forward_hook(feature_map_temp)
                            hook_list.append(handler)

                elif args.arch == 'resnet_50':
                    cov_layer = eval('model.maxpool')
                    handler = cov_layer.register_forward_hook(feature_map_temp)
                    hook_list.append(handler)
                    # ResNet50 per bottleneck
                    for i in range(4):
                        block = eval('model.layer%d' % (i + 1))
                        for j in range(model.num_blocks[i]):
                            cov_layer = block[j].relu1
                            handler = cov_layer.register_forward_hook(feature_map_temp)
                            hook_list.append(handler)
                            cov_layer = block[j].relu2
                            handler = cov_layer.register_forward_hook(feature_map_temp)
                            hook_list.append(handler)
                            cov_layer = block[j].relu3
                            handler = cov_layer.register_forward_hook(feature_map_temp)
                            hook_list.append(handler)
                            if j == 0:
                                cov_layer = block[j].relu3
                                handler = cov_layer.register_forward_hook(feature_map_temp)
                                hook_list.append(handler)
                #################################################

                pred = model(data)
                correct_in_batch = torch.argmax(pred, dim=1).eq(target.data).sum()
                correct_count_in_train_set += correct_in_batch
                for i in range(len(feature_map_box)):
                    coco_matrix = channel_correlation_matrix_GPU(feature_map_box[i])
                    if batch_idx == 0:
                        vis.heatmap(coco_matrix, win='Layer{}'.format(i + 1),
                                    opts=dict(title='Layer{}'.format(i + 1)))
                    file_path = args.result_dir + '/CCM_for_layers/' + 'batch_{}/'.format(batch_idx)
                    file_name = 'CCM_for_layer{}'.format(i + 1)
                    save_coco_M(coco_matrix, file_path, file_name)

                # release hook
                for item in hook_list:
                    item.remove()
                feature_map_box = []


if __name__ == '__main__':
    main()
