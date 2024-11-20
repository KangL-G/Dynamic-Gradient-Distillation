import numpy as np
import torch
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import random
import dgl
from ogb.nodeproppred import Evaluator


def distillation(student, teacher_scores, temp):
    return nn.KLDivLoss()(F.log_softmax(student / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp)

def L2_loss(y_pre,y_true):
    return torch.sum(torch.square(y_true-y_pre))/len(y_pre)

def accuracy(output, labels,details=False, hop_idx=None):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()
    if details:
        hop_num = np.bincount(hop_idx, minlength=7)
        true_idx = np.array((correct > 0).nonzero().squeeze(dim=1).cpu())
        true_hop = np.bincount(hop_idx[true_idx], minlength=7)/hop_num
        return result / len(labels), true_hop

    # mask = mask.float().cuda()
    # mask = mask / mask.mean()
    # correct *= mask
    # acc = correct.mean()
    acc = result / len(labels)
    return acc

def ogbn_acc(dataset, out, labels):
    ogb_evaluator = Evaluator(dataset)
    pred = out.argmax(1, keepdim=True)
    input_dict = {"y_true": labels.unsqueeze(1), "y_pred": pred}
    acc = ogb_evaluator.eval(input_dict)["acc"]
    return acc

def F1(output, labels):
    y_true = labels.detach().cpu().numpy()
    y_pred = output.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

def multi_label_auc(true, scores):
    total_auc = 0.
    y_scores = softmax(scores)
    if len(np.unique(true)) != y_scores.shape[1]:
        auc = roc_auc_score(true, y_scores, multi_class='ovo', labels=np.linspace(0, y_scores.shape[1]-1, y_scores.shape[1]))
    else:
        auc = roc_auc_score(true, y_scores, multi_class='ovo')

    return auc


def softmax(x):
    """
    对输入x的每一行计算softmax。

    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。

    代码利用softmax函数的性质: softmax(x) = softmax(x + c)

    参数:
    x -- 一个N维向量，或者M x N维numpy矩阵.

    返回值:
    x -- 在函数内部处理后的x
    """
    orig_shape = x.shape

    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出。 shape为(x.shape[0],)
        x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
        x = np.exp(x)  # 计算所有值的指数
        tmp = np.sum(x, axis=1)  # 每行求和
        x /= tmp.reshape((x.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求指数
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x


def eucli_dist(output, target):
    return torch.sqrt(torch.sum(torch.pow((output - target), 2)))




def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)