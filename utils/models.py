import os

from models.GCN import GCN
from models.GAT import GAT
from models.GraphSAGE import GraphSAGE
from models.MLP import MLP
from models.APPNP import APPNP
from models.MoNet import MoNet
from models.GCNII import GCNII
from dgl.nn.pytorch.conv import SGConv
import torch.nn.functional as F
import torch

def choose_teacher_model(param, G,labels):
    if param['teacher'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=G.ndata['feat'].shape[1],
            n_hidden=param['teacher_hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=param['teacher_layers'],
            activation=F.relu,
            dropout=param['dropout_t']).to(param['device'])
    elif param['teacher'] in ['GAT', 'SGAT']:
        model = GAT(g=G,
                    num_layers=param['teacher_layers'],
                    input_dim=G.ndata['feat'].shape[1],
                    hidden_dim=param['teacher_hidden'],
                    output_dim=labels.max().item() + 1,
                    activation=F.relu,
                    dropout_ratio=param['dropout_t'],
                    negative_slope=0.2,     # negative slope of leaky relu
                    residual=False).to(param['device'])
    elif param['teacher'] == 'GraphSAGE':
        model = GraphSAGE(g=G,
	         in_feats=G.ndata['feat'].shape[1],
                          n_hidden=param['teacher_hidden'],
                          n_classes=labels.max().item() + 1,
                          n_layers=param['teacher_layers'],
                          activation=F.relu,
                          dropout=param['dropout_t'],
                          aggregator_type='gcn').to(param['device'])
    elif param['teacher'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=G.ndata['feat'].shape[1],
                      hiddens=[64],
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,
                      k=10).to(param['device'])
    elif param['teacher'] == 'MoNet':
        model = MoNet(g=G,
                      in_feats=G.ndata['feat'].shape[1],
                      n_hidden=64,
                      out_feats=labels.max().item() + 1,
                      n_layers=1,
                      dim=2,
                      n_kernels=3,
                      dropout=0.7).to(param['device'])
    elif param['teacher'] == 'SGC':
        model = SGConv(in_feats=G.ndata['feat'].shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(param['device'])
    elif param['teacher'] == 'GCNII':
        if param['dataset'] == 'citeseer':
            param['layer'] = 32
            param['hidden'] = 256
            param['lamda'] = 0.6
            param['dropout_t'] = 0.7
        elif param['dataset'] == 'pubmed':
            param['hidden'] = 256
            param['lamda'] = 0.4
            param['dropout_t'] = 0.5
        model = GCNII(nfeat=G.ndata['feat'].shape[1],
                      nlayers=param['teacher_layers'],
                      nhidden=param['teacher_hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=param['dropout_t'],
                      lamda=param['lamda'],
                      alpha=param['alpha'],
                      variant=False).to(param['device'])
    return model

def choose_student_model(param,G,labels):
    if param['student'] == 'GCN':
        model = GCN(
            g=G,
            in_feats=G.ndata['feat'].shape[1],
            n_hidden=param['student_hidden'],
            n_classes=labels.max().item() + 1,
            n_layers=param['student_layers'],
            activation=F.relu,
            dropout=param['dropout_s']).to(param['device'])
    elif param['student'] == 'MLP':
        model = MLP(
            num_layers=param['student_layers'],
            input_dim=G.ndata['feat'].shape[1],
            hidden_dim=param['student_hidden'],
            output_dim=labels.max().item() + 1,
            dropout=param['dropout_s'],
            norm_type=param['norm_type']).to(param['device'])
    elif param['student'] in ['GAT', 'SGAT']:
        model = GAT(g=G,
                    num_layers=param['student_layers'],
                    input_dim=G.ndata['feat'].shape[1],
                    hidden_dim=param['student_hidden'],
                    output_dim=labels.max().item() + 1,
                    activation=F.relu,
                    dropout_ratio=0.6,
                    negative_slope=0.2,  # negative slope of leaky relu
                    residual=False).to(param['device'])
    elif param['student'] == 'GraphSAGE':
        model = GraphSAGE(in_feats=G.ndata['feat'].shape[1],
                          n_hidden=param['embed_dim'],
                          n_classes=labels.max().item() + 1,
                          n_layers=2,
                          activation=F.relu,
                          dropout=0.5,
                          aggregator_type=param['agg_type']).to(param['device'])
    elif param['student'] == 'APPNP':
        model = APPNP(g=G,
                      in_feats=G.ndata['feat'].shape[1],
                      hiddens=[64],
                      n_classes=labels.max().item() + 1,
                      activation=F.relu,
                      feat_drop=0.5,
                      edge_drop=0.5,
                      alpha=0.1,
                      k=10).to(param['device'])
    elif param['student'] == 'MoNet':
        model = MoNet(g=G,
                      in_feats=G.ndata['feat'].shape[1],
                      n_hidden=64,
                      out_feats=labels.max().item() + 1,
                      n_layers=1,
                      dim=2,
                      n_kernels=3,
                      dropout=0.7).to(param['device'])
    elif param['student'] == 'SGC':
        model = SGConv(in_feats=G.ndata['feat'].shape[1],
                       out_feats=labels.max().item() + 1,
                       k=2,
                       cached=True,
                       bias=False).to(param['device'])
    elif param['student'] == 'GCNII':
        if param['dataset'] == 'citeseer':
            param['layer'] = 32
            param['hidden'] = 256
            param['lamda'] = 0.6
            param['dropout_s'] = 0.7
        elif param['dataset'] == 'pubmed':
            param['hidden'] = 256
            param['lamda'] = 0.4
            param['dropout_s'] = 0.5
        model = GCNII(nfeat=G.ndata['feat'].shape[1],
                      nlayers=param['layer'],
                      nhidden=param['hidden'],
                      nclass=labels.max().item() + 1,
                      dropout=param['dropout_s'],
                      lamda=param['lamda'],
                      alpha=param['alpha'],
                      variant=False).to(param['device'])
    return model


def save_checkpoint(model, path):
    '''Saves model
    '''
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), path)
    print(f"save model to {path}")