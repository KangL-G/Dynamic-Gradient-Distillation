import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.nn import Parameter

# based on GAIL: State-->the teacher's outputs at each layer
#                Action-->the teacher's gradient at each layer


# 1、Grad --- Teacher Gradient Variation Generator(Time and Space)
'''
num_layers: Stay in tune with teachers
input_dim: the number of the nodes ---- state_dim
hidden_dim: Stay in tune with student's hidden_dim ---- action_dim
output_dim: the number of Class ---- action_dim
input: feature
output: action i.e. grad
objective: Each layer gets the gradient value by inputting the teacher's output
'''
class Grad_gen(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(Grad_gen, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear_or_not = True
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=1 - dropout)
        self.module_list = nn.ModuleList()
        self.log_std1 = Parameter(torch.zeros(hidden_dim))
        self.log_std2 = Parameter(torch.zeros(output_dim))

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model ()
            self.module_list.append(nn.Linear(input_dim, output_dim))
        else:
            self.linear_or_not = False
            self.module_list.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):  # num_layers-2 because first and last layers are already considered
                self.module_list.append(nn.Linear(input_dim, hidden_dim))
                # Last layer from last hidden dimension to output
            self.module_list.append(nn.Linear(input_dim, output_dim))

        for layer in self.module_list:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state, mode, select=False):
        # state = state[:len(state)-1]
        grad = []
        distb = []
        if select:
            if mode == 'stu':
                for j, state_tensor in enumerate(state):
                    for i in range(len(self.module_list)-2):
                        mean = F.tanh(self.module_list[i](self.dropout(state_tensor.t())))
                        std = torch.exp(self.log_std2 if i == len(self.module_list) - 1 else self.log_std1)
                        cov_mtx = torch.eye(self.output_dim if i == len(self.module_list) - 1 else self.hidden_dim).cuda() * (std ** 2)
                        cov_mtx = torch.eye(self.output_dim if i == len(self.module_list) - 1 else self.hidden_dim).cuda() * (std ** 2)
                        distb.append(MultivariateNormal(mean, cov_mtx))
                        grad.append(distb[-1].sample())
        else:
            if mode == 'tea':
                for i, state_tensor in enumerate(state):
                    mean = F.tanh(self.module_list[i](self.dropout(state_tensor.t())))
                    std = torch.exp(self.log_std2 if i == len(state) - 1 else self.log_std1)
                    cov_mtx = torch.eye(self.output_dim if i == len(state) - 1 else self.hidden_dim).cuda() * (std ** 2)
                    distb.append(MultivariateNormal(mean, cov_mtx))
                    grad.append(distb[i].sample())
            elif mode == 'stu':
                for j, state_tensor in enumerate(state):
                    for i in range(len(self.module_list)):
                        mean = F.tanh(self.module_list[i](self.dropout(state_tensor.t())))
                        std = torch.exp(self.log_std2 if i == len(self.module_list) - 1 else self.log_std1)
                        cov_mtx = torch.eye(self.output_dim if i == len(self.module_list) - 1 else self.hidden_dim).cuda() * (std ** 2)
                        cov_mtx = torch.eye(self.output_dim if i == len(self.module_list) - 1 else self.hidden_dim).cuda() * (std ** 2)
                        distb.append(MultivariateNormal(mean, cov_mtx))
                        grad.append(distb[-1].sample())

        return grad, distb

# 2、Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.linear_or_not = True
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=1 - dropout)
        self.module_list = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model ()
            self.module_list.append(nn.Linear(input_dim + output_dim, output_dim))
        else:
            self.linear_or_not = False
            self.module_list.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
            for _ in range(num_layers - 2):  # num_layers-2 because first and last layers are already considered
                self.module_list.append(nn.Linear(input_dim + hidden_dim, hidden_dim))
                # Last layer from last hidden dimension to output
            self.module_list.append(nn.Linear(input_dim + output_dim, output_dim))

        for layer in self.module_list:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, feat, gradient):
        scores = []
        if len(gradient) == self.num_layers:
            feat = feat[:self.num_layers]
            for i in range(len(gradient)):
                if gradient[i].shape[1] < self.hidden_dim and gradient[i].shape[1] > self.output_dim:
                    grad = torch.cat([gradient[i]]*(self.hidden_dim // gradient[i].shape[1]), dim=1)
                else:
                    grad = gradient[i]
                if feat[i].shape[1] ==grad.shape[1]:
                    sa = torch.cat(([feat[i], grad]), dim=0)
                    sa = sa.t()
                else:
                    sa = torch.cat((feat[i].t(), grad), dim=-1)
                scores.append(F.sigmoid(self.module_list[i](self.dropout(sa))))
        else:
            feat = feat[1:self.num_layers]
            for i in range(len(gradient)):
                if gradient[i].shape[1] < self.hidden_dim and gradient[i].shape[1] > self.output_dim:
                    grad = torch.cat([gradient[i]]*(self.hidden_dim // gradient[i].shape[1]), dim=1)
                else:
                    grad = gradient[i]
                if feat[i].shape[1] ==grad.shape[1]:
                    sa = torch.cat(([feat[i], grad]), dim=0)
                    sa = sa.t()
                else:
                    sa = torch.cat((feat[i].t(), grad), dim=-1)
                scores.append(F.sigmoid(self.module_list[i+1](self.dropout(sa))))

        return scores