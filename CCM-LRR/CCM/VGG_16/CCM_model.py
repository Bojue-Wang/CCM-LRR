import torch
import os
import numpy as np


def channel_correlation_matrix_GPU(feature_map):
    assert len(feature_map.size()) == 4
    sum_even_of_feature_map = torch.sum(feature_map, dim=0) / feature_map.size()[0]
    flatten_of_even_feature_map = sum_even_of_feature_map.view(sum_even_of_feature_map.shape[0], -1)
    even_of_each_victor = (
            torch.sum(flatten_of_even_feature_map, dim=1) / flatten_of_even_feature_map.shape[1]).unsqueeze(1)
    flatten_of_even_feature_map_minus_even_of_the_map = flatten_of_even_feature_map - even_of_each_victor
    COV = torch.matmul(flatten_of_even_feature_map_minus_even_of_the_map,
                       flatten_of_even_feature_map_minus_even_of_the_map.T)
    sqrt_sum_cube = torch.sqrt(torch.sum(flatten_of_even_feature_map_minus_even_of_the_map ** 2, dim=1)).unsqueeze(0)
    denominator = torch.matmul(sqrt_sum_cube.T, sqrt_sum_cube)
    co_co_matrix = COV / denominator
    return torch.abs(co_co_matrix)


def loss_per_layer(feature_map):
    co_co_matrix = channel_correlation_matrix_GPU(feature_map)
    saturation_index = torch.sum(co_co_matrix) / torch.numel(co_co_matrix)
    return saturation_index


def channel_correlation_matrix_GPU_no_abs(feature_map):
    assert len(feature_map.size()) == 4
    sum_even_of_feature_map = torch.sum(feature_map, dim=0) / feature_map.size()[0]
    flatten_of_even_feature_map = sum_even_of_feature_map.view(sum_even_of_feature_map.shape[0], -1)
    even_of_each_victor = (
            torch.sum(flatten_of_even_feature_map, dim=1) / flatten_of_even_feature_map.shape[1]).unsqueeze(1)
    flatten_of_even_feature_map_minus_even_of_the_map = flatten_of_even_feature_map - even_of_each_victor
    COV = torch.matmul(flatten_of_even_feature_map_minus_even_of_the_map,
                       flatten_of_even_feature_map_minus_even_of_the_map.T)
    sqrt_sum_cube = torch.sqrt(torch.sum(flatten_of_even_feature_map_minus_even_of_the_map ** 2, dim=1)).unsqueeze(0)
    denominator = torch.matmul(sqrt_sum_cube.T, sqrt_sum_cube)
    co_co_matrix = COV / denominator
    return co_co_matrix


def save_coco_M(tensor_to_save, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    save_file = os.path.join(file_path, "{}.pth".format(file_name))
    torch.save(tensor_to_save, save_file)
    print("{} saved".format(file_name))


def load_coco_M(file_path, file_name):
    save_file = os.path.join(file_path, "{}.pth".format(file_name))
    if not save_file:
        print("Can not find file!")
        return
    co_co_M = torch.load(save_file)
    return co_co_M


test_feature = torch.rand(16, 3, 4, 4)
print(test_feature.size())

sum_of_feature_map = torch.sum(test_feature, dim=0)
print(sum_of_feature_map.size())

sum_even_of_feature_map = sum_of_feature_map / test_feature.size()[0]
print(sum_even_of_feature_map.size())  # 样本均值

flatten_of_even_feature_map = sum_even_of_feature_map.view(sum_even_of_feature_map.shape[0], -1)
print(flatten_of_even_feature_map.size())       # 按通道打平

even_of_each_victor = (
            torch.sum(flatten_of_even_feature_map, dim=1) / flatten_of_even_feature_map.shape[1]).unsqueeze(1)
# 求通道均值

print(torch.sum(flatten_of_even_feature_map, dim=1).size())
print(flatten_of_even_feature_map.shape[1])
print(even_of_each_victor.size())
# calculate even of each channel

flatten_of_even_feature_map_minus_even_of_the_map = flatten_of_even_feature_map - even_of_each_victor   # X-E(X)
