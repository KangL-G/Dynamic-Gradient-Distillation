import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
from torch import nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

import dgl
import argparse
import time
import matplotlib.pyplot as plt
import collections
import random
import datetime


from pathlib import Path

from utils.plot_utils import parameters
from utils.metrics import *
from utils.TSGF import gradient_fusion


def DGD(param, s_middle_f, student_model, grad_generator):
    import ipdb
    ipdb.set_trace()

    middle_feat_s = []
    for i in range(param['student_layers']):
        middle_feat_s.append(s_middle_f[i].detach().cpu())

    grad_fused = []
    

    grad_s = [copy.deepcopy(student_model.layers[i].weight.grad.detach().cpu().clone()) for i in range(len(student_model.layers))]

    grad_gen, distb = grad_generator(middle_feat_s, param['mode'])

    # split data
    num = len(grad_gen)//len(student_model.layers)
    # 为了确保所有数据都被分配，计算可能未被均匀分配的剩余数据项
    remainder = len(grad_gen) % len(student_model.layers)
    layered_data = {}  # 使用字典来保存每层的数据
    start_index = 0
    for i in range(len(student_model.layers)):
        if i == len(student_model.layers) - 1:
            layer = grad_gen[start_index:]
        else:
            end_index = start_index + num + (1 if remainder > 0 else 0)
            layer = grad_gen[start_index:end_index]
            start_index = end_index
            remainder -= 1 if remainder > 0 else 0  # 减少余数计数

        # 将每层数据存储到字典中
        layered_data[f'layer_{i + 1}'] = layer

    for i in range(len(student_model.layers)):
        grad_g = layered_data[f'layer_{i + 1}']
        grad_fused.append(dtw_dsasm(grad_g, grad_s[i], param))
        
    return grad_fused