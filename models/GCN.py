import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()

        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        if n_layers == 1:
            self.layers.append(GraphConv(in_feats, n_classes, activation=activation))
        else:
            self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
            for i in range(n_layers - 2):
                self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
            self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout =dropout
        self.reset_parameters()

    def reset_parameters(self):
        for i in self.layers:
            nn.init.kaiming_normal_(i.weight)
            nn.init.constant_(i.bias, 0.0)


    def forward(self, features):
        h = features
        middle_feats = []
        middle_feats.append(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = F.dropout(h, self.dropout, training=self.training)
            h = layer(self.g, h)
            middle_feats.append(h)
        return h, middle_feats


class ogb_GCN(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout, use_linear=False):
        super().__init__()
        self.g = g
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.bns.append(nn.BatchNorm1d(out_hidden))

        self.dropout0 = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, feat):
        h = feat
        h = self.dropout0(h)
        for i in range(self.n_layers):
            conv = self.convs[i](self.g, h)
            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv
            if i < self.n_layers - 1:
                h = self.bns[i](h)
                h = self.activation(h)
                h = self.dropout(h)
        return h
