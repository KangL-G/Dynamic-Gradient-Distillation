import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA


def expand_to_same_dim(data, target_dim):
    if data.shape[1] == target_dim:
        return data
    elif data.shape[1] < target_dim:
        expanded_data = torch.zeros((data.shape[0], target_dim))
        #expanded_data = torch.zeros((data.shape[0], target_dim)).cuda()
        expanded_data[:, :data.shape[1]] = data
        return expanded_data
    else:
        pca = PCA(n_components=target_dim)
        expanded_data = torch.Tensor(pca.fit_transform(data.detach().cpu().numpy()))
        #expanded_data = torch.Tensor(pca.fit_transform(data.detach().cpu().numpy())).cuda()
        return expanded_data

def simple_self_attention(inputs, grad_s, param):
    d_k = grad_s.shape[-1]
    for data in inputs:
        if param['student'] == 'GCN':
            fused_data = torch.zeros_like(grad_s).cuda()
            expanded_data = expand_to_same_dim(data, grad_s.shape[1])
            queries = expanded_data
            values = grad_s
            keys = grad_s
            scores = (queries @ keys.T) / (d_k ** 0.5)
            weights = F.softmax(scores, dim=-1)
            fused_data += weights @ values
        elif param['student'] == 'MLP':
            fused_data = torch.zeros_like(grad_s.t())
            #fused_data = torch.zeros_like(grad_s.t()).cuda()
            expanded_data = expand_to_same_dim(data, grad_s.t().shape[1])
            queries = expanded_data
            values = grad_s.t()
            keys = grad_s.t()
            scores = (queries @ keys.T) / (d_k ** 0.5)
            weights = F.softmax(scores, dim=-1)
            fused_data += weights @ values

    # output = torch.matmul(attention_weights, values)
    return fused_data

def gradient_fusion(grad_gen, grad_s, param):

    if param['student'] == 'GCN':
        fused_grad = simple_self_attention(grad_gen, grad_s, param)
    elif param['student'] == 'MLP':
        fused_grad = (simple_self_attention(grad_gen, grad_s, param)).t()
    # loss_distance = combined_loss(fused_grad, grad_s, alpha)

    return fused_grad

