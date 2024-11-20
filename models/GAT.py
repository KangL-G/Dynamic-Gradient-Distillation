import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

class GAT(nn.Module):
    def __init__(
        self, g, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation, num_heads=4, attn_drop=0.3, negative_slope=0.2, residual=False):
        super(GAT, self).__init__()

        self.g = g
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        heads = ([num_heads] * num_layers) + [1]

        self.layers.append(GATConv(input_dim, hidden_dim, heads[0], dropout_ratio, attn_drop, negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads[l-1], hidden_dim, heads[l], dropout_ratio, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(hidden_dim * heads[-2], output_dim, heads[-1], dropout_ratio, attn_drop, negative_slope, residual, None))

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.layers:
            nn.init.kaiming_normal_(i.fc.weight)
            nn.init.constant_(i.bias, 0.0)

    def forward(self, feats):
        h = feats
        h_list = []
        h_list.append(h)

        for l, layer in enumerate(self.layers):
            h = layer(self.g, h)
            if l != self.num_layers - 1:
                h = h.flatten(1)
            else:
                h = h.mean(1)
            h_list.append(h)

        return h, h_list