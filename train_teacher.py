import csv
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

import dgl
import argparse
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
import collections
import random
import datetime
import gc

from models.utils import get_training_config, load_checkpoint
from pathlib import Path

from utils.logger import get_logger, check_writable
from utils.models import save_checkpoint, choose_teacher_model, choose_student_model
from utils.metrics import accuracy, F1, multi_label_auc
import heapq
from memory_profiler import profile

# 动态调整权重函数
def adjust_lambda_grad(epoch, param, initial_lambda=0.9):
    return initial_lambda ** (epoch)
    

def train_teacher(param, teacher_model, optimizer,  grad_generator, g_optimizer, grad_Discriminator, d_optimizer, data, G,  label):
    train, valid, test = data
    dur = []
    t0 = time.time()

    best = 0
    es = 0
    grad_gen_states = []

    # define loss function
    loss_CE = torch.nn.NLLLoss()
    feat = G.ndata['feat'].to(param['device'])
    
    if param['dataset'] =="ogbn-products":
        G = dgl.add_self_loop(G)
    
    teacher_model.g = G
    for layer in teacher_model.layers:
        layer.g = G

    for epoch in range(param['max_epoch']):
        d_loss_real = 0.
        d_loss_fake = 0.
        loss_g = 0.
        d_loss = 0.
        loss_disc = 0.

        teacher_model.train()
        grad_generator.train()
        grad_Discriminator.train()

        # Save the previous gradient set
        if epoch > 0:
            weight_grad_pre = weight_grad_now
  

        if param['teacher'] in ['GCN', 'APPNP']:
            output_t, mid_t = teacher_model(feat)
        elif param['teacher'] in ['GAT', 'SGAT']:
            output_t, mid_t = teacher_model(feat)
        elif param['teacher'] in ['GraphSAGE', 'SGC']:
            output_t = teacher_model(G, feat)
        elif param['teacher'] == 'MoNet':
            us, vs = G.edges(order='eid')
            udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
            pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
            output_t, _ = teacher_model(G.ndata['feat'], pseudo)
        else:
            raise ValueError(f'Undefined Model')

        loss = loss_CE(output_t.log_softmax(dim=1)[train], label[train])

        optimizer.zero_grad()
        loss.backward()
        middle_feat_t = []
        weight_grad_now = []
        for i in range(len(teacher_model.layers)):
            if param['teacher'] == 'GCN':
                weight_grad_now.append(copy.deepcopy(teacher_model.layers[i].weight.grad.detach().clone()))
            elif param['teacher'] == 'GraphSAGE':
                weight_grad_now.append(copy.deepcopy(teacher_model.layers[i].fc_neigh.weight.grad.detach().clone()))
            elif param['teacher'] == 'GAT':
                weight_grad_now.append(copy.deepcopy(teacher_model.layers[i].fc.weight.grad.detach().clone()))
            middle_feat_t.append(mid_t[i].detach())
        
        
        optimizer.step()

        # train generator and discriminator
        if epoch > 0 and param['K'] != 0:
            # Generate Behavior
            generated_grad, distb = grad_generator(middle_feat_t, param['mode'])
            # different epoch, same layer
            real_action1 = [torch.cat([weight_grad_now[i].detach()]) for i in range(len(teacher_model.layers))]
            # different layer, same epoch
            real_action2 = [torch.cat([weight_grad_pre[i+1].detach()]) for i in range(len(teacher_model.layers)-1)]

            fake_action = [torch.cat([generated_grad[i].detach()]) for i in range(len(teacher_model.layers))]


            # stage 1: different epoch, same layer
            grad_real1 = grad_Discriminator(middle_feat_t, real_action1)
            grad_fake = grad_Discriminator(middle_feat_t, fake_action)

            # stage 2: different layer, same epoch
            grad_real2 = grad_Discriminator(middle_feat_t, real_action2)

            for i in range(len(teacher_model.layers)):
                d_loss_real += F.mse_loss(grad_real1[i], torch.zeros_like(grad_real1[i]))
                d_loss_fake += F.mse_loss(grad_fake[i], torch.ones_like(grad_fake[i]))

            for i in range(len(teacher_model.layers)-1):
                d_loss_real += F.mse_loss(grad_real2[i], torch.zeros_like(grad_real2[i]))
                d_loss_fake += F.mse_loss(grad_fake[i+1], torch.ones_like(grad_fake[i+1]))
            d_loss += (d_loss_real + d_loss_fake) / 2

            loss_disc = d_loss / len(teacher_model.layers)

            d_optimizer.zero_grad()
            loss_disc.backward(retain_graph=True)
            d_optimizer.step()



            for i in range(len(distb)):
                loss_g += ((-1) * adjust_lambda_grad(epoch, param)
                           * distb[i].log_prob(generated_grad[i])).mean()
            loss_g = loss_g/len(distb)


            g_optimizer.zero_grad()
            loss_g.backward(retain_graph=True)
            g_optimizer.step()

        loss_data = np.array(loss.item())

        dur.append(time.time() - t0)
        train_acc, _, _ = evalution(param, teacher_model, train, G, label)
        val_acc, _, _ = evalution(param, teacher_model, valid, G, label)
        

        print(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) +
              'Epoch %d | Loss: %.4f | Loss_d: %.4f | Loss_g: %.4f | train_acc: %.4f | val_acc: %.4f | Time(s) %.4f'
              % (epoch, loss_data.item(), loss_disc, loss_g, train_acc.item(), val_acc.item(), dur[-1]))


        if param['K'] != 0:
            state_dict = grad_generator.state_dict()
            if len(grad_gen_states) < param['K']:
            # 压缩模型参数
                heapq.heappush(grad_gen_states, (val_acc, epoch, state_dict))
            else:
              # 如果当前 val_acc 大于堆中最小 val_acc，则替换
                if val_acc > grad_gen_states[0][0]:
                    heapq.heapreplace(grad_gen_states, (val_acc, epoch, state_dict))


        if val_acc > best:
            best = val_acc
            state = dict([('model', copy.deepcopy(teacher_model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            es = 0
        else:
            es += 1
        if epoch == param['max_epoch'] or es == 30:
            print("Stop!!!")
            break
        
    
    teacher_model.load_state_dict(state['model'])

    test_score, auc_score, f1_score = evalution(param, teacher_model, test, G, label)

    # Weighted Ensemble Generator
    #integ_param = {}
    #for item in grad_gen_states:
        #weight = item[0]
        #grad_gen_dict = item[2]
        #for key, value in grad_gen_dict.items():
            #if key not in integ_param:
                #integ_param[key] = torch.zeros_like(value)
            #integ_param[key] += weight * value
            
        # Weighted Ensemble Generator
    integ_param = {}

    for item in grad_gen_states:
        weight = item[0]
        grad_gen_dict = item[2]

        # 将参数移回 GPU 上
        for key, value in grad_gen_dict.items():
            value = value.to(param['device'])  # 将参数移回 GPU

            if key not in integ_param:
                integ_param[key] = torch.zeros_like(value)
            integ_param[key] += weight * value

    with torch.no_grad():  # 确保在修改参数时不会计算梯度
        for name, param in grad_generator.named_parameters():
            if name in integ_param:
                print(f"Updating {name}")
                param.copy_(integ_param[name])  # 使用 copy_ 方法来替换参数值
    
    
    
    print(str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())) + " The teacher's test set results: acc_test= %.4f" % (test_score))
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(dur[-1]))

    return test_score



def evalution(param, teacher_model, data, G,  label):
    teacher_model.eval()
    with torch.no_grad():
        feat = G.ndata['feat'].to(param['device'])
        if param['dataset'] =="ogbn-products":
            G = dgl.add_self_loop(G)

        teacher_model.g = G
        for layer in teacher_model.layers:
            layer.g = G

        if param['teacher'] in ['GCN', 'APPNP']:
            output_t, _= teacher_model(feat)
        elif param['teacher'] in ['GAT', 'SGAT']:
            output_t, _ = teacher_model(feat)
        elif param['teacher'] in ['GraphSAGE', 'SGC']:
            output_t= teacher_model(G, feat)
        elif param['teacher'] == 'MoNet':
            us, vs = G.edges(order='eid')
            udeg, vdeg = 1 / torch.sqrt(G.in_degrees(us).float()), 1 / torch.sqrt(G.in_degrees(vs).float())
            pseudo = torch.cat([udeg.unsqueeze(1), vdeg.unsqueeze(1)], dim=1)
            output_t = teacher_model(G.ndata['feat'], pseudo)
        elif param['teacher'] == 'GCNII':
            output_t = teacher_model(feat, adj)
        else:
            raise ValueError(f'Undefined Model')
        logp = F.log_softmax(output_t, dim=1)

        acc = accuracy(logp[data], label[data])
        auc = multi_label_auc(label[data].detach().cpu().numpy(), logp[data].detach().cpu().numpy())
        f1 = F1(logp[data], label[data])

    return acc, auc, f1
