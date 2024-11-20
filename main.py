import json
import os
import numpy as np
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
import dgl
import argparse
import time
import nni
import csv

#from data.get_dataset import get_experiment_config
from models.utils import load_checkpoint
from pathlib import Path

from dataloader import *
from train_student import train_student
from train_teacher import train_teacher, evalution

from utils.logger import check_writable
from utils.metrics import set_seed
from utils.models import choose_teacher_model, save_checkpoint, choose_student_model
from models import Gradient_Agent

import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cpu')

def main(param, teacher_model, student_model, grad_generator, grad_Discriminator, t_optimizer, g_optimizer, a_optimizer, s_optimizer, data, g, features, labels, model_tea_path, model_path):

    if param['mode'] == 'stu':
        print("############ Student model with Teacher #############")
        #t_score, _, _ = evalution(param, teacher_model, idx_test, g, labels)
        #print(str(time.strftime("[%Y-%m-%d %H:%M:%S]",time.localtime())) + " The teacher's test set results: acc_test= %.4f" % (t_score))
        t_score = 0
        test_acc, test_val, test_best, epoch = train_student(param, student_model, teacher_model, grad_generator,
                                                              s_optimizer, data, g, features, labels)
        return t_score, test_acc, test_val, test_best, epoch
        
    elif param['mode'] == 'tea':
        print("############ Teacher model#############")
        test_t = train_teacher(param, teacher_model, t_optimizer, grad_generator, g_optimizer, grad_Discriminator,
                                a_optimizer, data, g, labels)
        if param['K'] != 0:
            save_checkpoint(teacher_model, model_tea_path.joinpath('t_' + param['teacher'] + '_' + str(param['K']) + '.pt'))
            save_checkpoint(grad_generator, model_path.joinpath('g_' + 'generator' + '_' + str(param['K']) + '.pt'))
        else:
            save_checkpoint(teacher_model, model_tea_path.joinpath('t_' + param['teacher'] + '_' + str(param['K']) + '.pt'))
        return test_t
    else:
        raise ValueError("Missing mode settings!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument('--batch_size', type=str, default=256, help="batch size used for training, validation and test")
    parser.add_argument('--max_epoch', type=str, default=500)
    parser.add_argument('--teacher', type=str, default='GCN', help='Teacher Model')
    parser.add_argument('--teacher_hidden', type=int, default=256, help='Teacher Model hidden')
    parser.add_argument('--teacher_layers', type=int, default=4, help='Teacher Model layer')
    parser.add_argument('--student', type=str, default='MLP', help='student Model')
    parser.add_argument('--student_hidden', type=int, default=64, help='Student Model hidden')
    parser.add_argument('--student_layers', type=int, default=2, help='Student Model layers')
    parser.add_argument('--seed', type=int, default=2026, help="random seed")
    parser.add_argument('--mode', type=str, default='stu', help="select student or teacher for grad_generator")
    parser.add_argument('--K', type=int, default=4, help="select the number of grad generator")
    parser.add_argument('--temp', type=float, default=4.0, help="when student is MLP, the value of temperature")
    parser.add_argument('--split_idx', type=int, default=0, help="For Non-Homo datasets only, one of [0,1,2,3,4]")
    parser.add_argument( "--labelrate_train",type=int,default=20,help="How many labeled data per class as train set",)
    parser.add_argument("--labelrate_val",type=int,default=30,help="How many labeled data per class in valid set",)

    args = parser.parse_args()
    param = args.__dict__
    param.update(nni.get_next_parameter())
    
    # load param
    if os.path.exists("./param/best_param.json"):
        param = json.loads(open("./param/best_param.json", 'r').read())[param['dataset']][param['teacher']][param['student']]
    param['device'] = device
    set_seed(param['seed'])
    print(param)
    
    # load data
    g, labels, idx_train, idx_val, idx_test = load_data(param['dataset'], dataset_path="./data",  split_idx=param['split_idx'], seed=param['seed'],labelrate_train=args.labelrate_train,
        labelrate_val=args.labelrate_val,)

    g = g.to(device)
    features = g.ndata["feat"].to(device)
    labels = labels.to(device)

    print('We have %d nodes.' % g.number_of_nodes())
    print('We have %d edges.' % g.number_of_edges())

    data = (idx_train, idx_val, idx_test)

    # choose model
    teacher_model = choose_teacher_model(param, g, labels)
    grad_generator = Gradient_Agent.Grad_gen(num_layers=param['teacher_layers'],
                                             input_dim=features.shape[0],
                                             hidden_dim=param['student_hidden'],
                                             output_dim=labels.max().item() + 1,
                                             dropout=param['dropout_s']).cpu()
    grad_Discriminator = Gradient_Agent.Discriminator(num_layers=param['teacher_layers'],
                                                      input_dim=features.shape[0],
                                                      hidden_dim=param['teacher_hidden'],
                                                      output_dim=labels.max().item() + 1,
                                                      dropout=param['dropout_s']).to(param['device'])
    student_model = choose_student_model(param, g, labels)

    t_optimizer = optim.Adam(teacher_model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    g_optimizer = optim.Adam(list(grad_generator.parameters()), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    a_optimizer = optim.Adam(list(grad_Discriminator.parameters()), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    s_optimizer = optim.Adam(list(student_model.parameters()), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    # load or train the teacher
    model_tea_path = Path.cwd().joinpath('outputs', param['dataset'], param['teacher'], 'tea')
    check_writable(model_tea_path)
    print(model_tea_path)
    
    if param['mode'] == 'stu':
        load_checkpoint(teacher_model, model_tea_path.joinpath('t_' + param['teacher'] + '.pt'), device)

    print(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
    

    test_acc_list = []
    test_val_list = []
    test_best_list = []
    test_teacher_list = []


    if param['mode'] == 'stu':
        for seed in range(3):
            param['seed'] += seed
            set_seed(param['seed'])
            print("*********************  The current param   *********************")
            print(param)
            model_path = Path.cwd().joinpath('outputs', param['dataset'], param['teacher'], str(param['K']))
            check_writable(model_path)

            if param['K'] != 0:
                load_checkpoint(grad_generator, model_path.joinpath('g_' + 'generator' + '_' + str(param['K']) + '.pt'), device)

            test_teacher, test_acc, test_val, test_best, final_epoch = main(param, teacher_model, student_model, grad_generator, grad_Discriminator, t_optimizer, g_optimizer, a_optimizer, s_optimizer, data, g, features, labels, model_tea_path, model_path)
            test_acc_list.append(test_acc)
            test_val_list.append(test_val)
            test_best_list.append(test_best)

            test_teacher_list.append(test_teacher)
            nni.report_intermediate_result(test_acc)

    elif param['mode'] == 'tea':
        for seed in range(1):
            param['seed'] += 0
            set_seed(param['seed'])
            model_path = Path.cwd().joinpath('outputs', param['dataset'], param['teacher'], str(param['K']))
            check_writable(model_path)

            test_teacher = main(param, teacher_model, student_model, grad_generator, grad_Discriminator, t_optimizer, g_optimizer, a_optimizer, s_optimizer, data, g, features, labels, model_tea_path, model_path)
            test_teacher_list.append(test_teacher.detach().cpu().numpy())
    else:
        raise ValueError("Missing mode settings!")

    nni.report_final_result(np.mean(test_acc_list))

    outFile = open((Path.cwd().joinpath('outputs', param['dataset'], param['teacher'])).joinpath('PerformMetrics.csv'), 'a+',
                   newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))]

    for v, k in param.items():
        results.append(v)
        results.append(k)
    if param['mode'] == 'tea':
        results.append(str(test_teacher_list))
    else:
        results.append(final_epoch)
        results.append(str(test_acc_list))
        results.append(str(test_val_list))
        results.append(str(test_best_list))
        results.append(str(test_teacher_list))

        results.append(str(np.mean(test_acc_list) * 100))
        results.append(str(np.mean(test_val_list) * 100))
        results.append(str(np.mean(test_best_list) * 100))
        #results.append(str(np.mean(test_teacher_list)))

        results.append(str(np.std(test_acc_list) * 100))
        results.append(str(np.std(test_val_list) * 100))
        results.append(str(np.std(test_best_list) * 100))
        #results.append(str(np.std(test_teacher_list)))
    writer.writerow(results)