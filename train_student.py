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

from models.utils import get_training_config, load_checkpoint
from pathlib import Path
from utils.logger import get_logger, check_writable
from utils.metrics import *
from DGD import *


def train_student(param, student_model, teacher_model, grad_generator, optimizer, data, G, feats, labels):
    
    train, valid, test = data
    device = param['device']
    dur = []
    t0 = time.time()

    best_model = None
    best = 0
    es = 0
    test_best = 0
    test_val = 0

    # teacher output
    teacher_model.eval()
    g_t = dgl.add_self_loop(G)
    teacher_model.g = g_t 
    for layer in teacher_model.layers:
        layer.g = g_t 
    with torch.no_grad():
        output_t, t_middle_f = teacher_model(feats)

    # define loss function
    loss_n = torch.nn.NLLLoss()
    loss_kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    idx_l = train
    idx_t = torch.cat([train, valid, test])
    batch_size = param["batch_size"]
    num_node = feats.shape[0]
    #edge_idx_list = extract_indices(G)
    #edge_idx = edge_idx_list[0]


    feats_l, labels_l = feats[idx_l], labels[idx_l]
    feats_t, out_t = feats[idx_t], output_t.log_softmax(dim=1)[idx_t]
    feats_val, labels_val = feats[valid], labels[valid]
    feats_test, labels_test = feats[test], labels[test]
    target_t = labels[idx_t]

    for epoch in range(1, param['max_epoch']+1):
        student_model.train()
        grad_generator.eval()

        # gen_grad = []
        loss_grad = torch.tensor(0.).cuda()
        loss_kd_total = torch.tensor(0.).cuda()

        if param['student'] =='GCN':
            student_model.g = G
            for layer in student_model.layers:
                layer.g = G
            output_s, s_middle_f = student_model(feats)
            loss_l = loss_n(output_s.log_softmax(dim=1)[train], labels[train])
            loss_t = F.mse_loss(output_s[train], output_t[train])
            optimizer.zero_grad()
            (loss_l+loss_t).backward()
            optimizer.step()
        elif param['student'] =='MLP':
            #num_batches = max(1, feats.shape[0] // batch_size)
            #idx_batch = torch.randperm(feats.shape[0])[: num_batches * batch_size]

            #if num_batches == 1:
                #idx_batch = idx_batch.view(1, -1)
            #else:
                #idx_batch = idx_batch.view(num_batches, batch_size)
                

            num_batches = int(np.ceil(len(feats) / param['batch_size']))

            total_loss = 0
            all_midd = []
            #for i in range(num_batches):
            #logits, s_middle_f = student_model(feats[batch_size * i: batch_size * (i + 1)]) 
            logits, s_middle_f = student_model(feats)
            out = logits.log_softmax(dim=1)
            loss_l = loss_n(out, labels)
            loss_t = loss_kl(out,out_t)

                #loss_l = loss_n(out, labels[batch_size * i: batch_size * (i + 1)])
                #loss_t = loss_kl(out,out_t[batch_size * i: batch_size * (i + 1)])
            loss = param['alpha'] * loss_l + (1-param['alpha']) * loss_t
               # total_loss += loss

            #total_loss = total_loss/num_batches


            if epoch > 1 and param['K'] != 0:
               # _, s_middle_f = student_model(feats)
                
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
                
            optimizer.zero_grad()
            loss.backward(create_graph=True)

            if epoch > 1 and param['beta'] != 0 and param['K'] != 0:
                grad_first = []
                grad_out = []
                for i in range(len(optimizer.param_groups[0]['params'])):
                    grad_first.append(optimizer.param_groups[0]['params'][i].grad)
                    grad_out.append(torch.ones_like(grad_first[i]).to(param['device']))
                second_grad = (torch.autograd.grad(grad_first,
                                               optimizer.param_groups[0]['params'],
                                               grad_outputs=grad_out,
                                               create_graph=False))
                for i in range(len(student_model.layers)):
                    student_model.layers[i].weight.grad = student_model.layers[i].weight.grad + \
                                                      param['beta'] * second_grad[2 * i+4] * \
                                                      (grad_fused[i].to(param['device']) - student_model.layers[i].weight.grad)

        optimizer.step()

        dur.append(time.time() - t0)

        train_score, val_score, test_acc = evalution(param, student_model, train, valid, test, feats, G, labels)
        
        
        # print(loss_grad)
        # print(student_model.layers[1].weight.grad)
        print(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) +
              'Epoch %d | Loss_ce: %.4f | Loss_kd: %.4f | Train_acc: %.4f | Val_acc: %.4f | Test_acc: %.4f || Val Best: %.4f | Test Val: %.4f | Test Best: %.4f | Time(s) %.4f'%
              (epoch,  loss_l, loss_t, train_score, val_score, test_acc, best, test_val, test_best, dur[-1]))
              
        if test_acc > test_best:
            test_best = test_acc
            
            
        if val_score >= best:
            best = val_score
            test_val = test_acc
            state = dict([('student_model', copy.deepcopy(student_model.state_dict()))])
            es = 0
        else:
            es += 1

        if epoch == param['max_epoch'] or es == 50:
            print("Early stopping!!!")
            break

    student_model.load_state_dict(state['student_model'])

    # flops_s, params_s = profile(student_model, (G.ndata['feat'].to(device),))
    # flops_t, params_t = profile(teacher_model, (G.ndata['feat'].to(device),))
    print("Optimization Finished!")
    print(f"Teacher model-params:{parameters(teacher_model)}")
    print(f"Student model-params:{parameters(student_model)}")
    return test_acc, test_val, test_best, epoch


def evalution(param, student_model, train, valid, test, feats, G, label):

    student_model.eval()
    with torch.no_grad():
        if param['student'] =='GCN':
            student_model.g = G
            for layer in student_model.layers:
                layer.g = G
        
        num_batches = int(np.ceil(len(feats) / param['batch_size']))
            
        output=[]
        for i in range(num_batches):
            logits, _ = student_model(feats[param['batch_size']*i:param['batch_size']*(i+1)])
            out = logits.log_softmax(dim=1)
            output.append(out)
        s_out = torch.cat(output)
          
        if param['dataset'] in ["cora", "citeseer", "pubmed", "coauthor-phy", "amazon-photo"]:
            acc = accuracy(out[data], label[data])
        elif param['dataset'] in ["ogbn-arxiv", "ogbn-products"]:
            train_acc = ogbn_acc(param['dataset'], s_out[train], label[train])
            val_acc = ogbn_acc(param['dataset'], s_out[valid], label[valid])
            test_acc = ogbn_acc(param['dataset'], s_out[test], label[test])
        test_auc = []
        test_f1 = []

        return train_acc, val_acc, test_acc



