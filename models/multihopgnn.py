from torch import nn
from torch.nn import init
import os
import sys
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from AGCA import *

class Interaction_GCN(nn.Module):
    def __init__(self, hidden_channels):
        super(Interaction_GCN, self).__init__()
        self.fc = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, inputs):
        x = inputs.mean(dim=1, keepdim=True)
        return self.fc(x)

class Interaction_SAGE(nn.Module):
    def __init__(self, hidden_channels):
        super(Interaction_SAGE, self).__init__()
        self.fc_l = nn.Linear(hidden_channels, hidden_channels)
        self.fc_r = nn.Linear(hidden_channels, hidden_channels)
    def forward(self, inputs):
        neighbor = inputs.mean(dim=1, keepdim=True)
        neighbor = self.fc_r(neighbor)
        x = self.fc_l(inputs)
        x = (x + neighbor)
        return x

class Interaction_Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=0.5, learnable=False, positive=False,
                 identity=False):
        super().__init__()
        self.scale = scale
        self.dual_norm = dual_norm
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

        # Always define scale_param but conditionally initialize it
        self.scale_param = nn.Parameter(torch.empty(dim))
        if learnable:
            import math
            scale_init = math.log(scale) if positive else scale
            self.scale_param.data.fill_(scale_init)

    def forward(self, x):
        if self.scale > 0.0:
            x_norm = nn.functional.normalize(x, dim=-1)
            # x_norm = x
            sim = torch.matmul(x_norm, x_norm.transpose(-2, -1)) / self.temp
            # sim = torch.matmul(x_norm.transpose(-2, -1), x_norm) / self.temp
            sim = nn.functional.softmax(sim, dim=-1)
            if self.dual_norm:
                sim += nn.functional.softmax(sim, dim=-2)
            x_neg = torch.matmul(sim, x)

            if self.learnable:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                x = x - scale.view(1, 1, -1) * x_neg
            else:
                x = x - self.scale * x_neg

            if self.identity:
                x += x

        return self.layernorm(x)
        # return x


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio):
        super(AGCA, self).__init__()
        self.hide_channel = in_channel // ratio

        # 将池化层改为1D（沿节点维度）
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channel, self.hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)

        # 注意力矩阵参数
        self.A0 = torch.eye(self.hide_channel).to('cuda')
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((self.hide_channel, self.hide_channel))),
                               requires_grad=True)
        init.constant_(self.A2, 1e-6)

        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(self.hide_channel, in_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj_matrix):

        N, C = x.size()
        y = self.avg_pool(x)  # (N, self.hide_channel)
        y = self.conv1(y)  # (N, self.hide_channel)
        y = y.view(N, self.hide_channel, 1)  # (N, H, 1)
        y = y.transpose(1, 2)  # (N, 1, H)
        y = y.expand(N, self.hide_channel, self.hide_channel)  # (N, H, H)

        A1 = self.conv2(y)  # (N, 1, H)
        A1 = self.softmax(A1)  # (N, 1, H)
        A1 = A1.squeeze(1)  # (N, H)

        A_adj = adj_matrix.unsqueeze(1)  # (N, 1, N)
        A_adj = A_adj @ A_adj.transpose(2, 3)  # (N, 1, N) × (N, 1, N) → (N, 1, N)

        A = (self.A0 * A1.unsqueeze(1)) + self.A2.unsqueeze(0)  # (H, H) + (1, H, H) → (1, H, H)
        A = A
        A = A / torch.sum(A, dim=2, keepdim=True).clamp(min_val=1e-9)

        y = y @ A  # (N, H, H) × (H, H) → (N, H, H)
        y = self.relu(self.conv3(y.squeeze(2)))  # (N, 1)
        y = y.view(N, self.hide_channel, 1)  # (N, H, 1)
        y = self.sigmoid(self.conv4(y))  # (N, C)

        return x * y  # (N, C)

class HopGNN(torch.nn.Module):
    def __init__(self, g, in_channels, hidden_channels, out_channels, num_hop=6, dropout=0.5, activation='relu',
                 feature_inter='attention', inter_layer=2, feature_fusion='attention', norm_type='ln'):
        super().__init__()
        self.num_hop = num_hop
        self.feature_inter_type = feature_inter
        self.feature_fusion = feature_fusion
        self.dropout = nn.Dropout(dropout)
        self.pre = False
        self.g = g
        self.norm_type = norm_type
        self.build_activation(activation)

        # ContraNorm
        self.contra_norm = ContraNorm(dim=hidden_channels)
        # AGCA module (加入AGCA)

        self.agca = AGCA(in_channel=in_channels, ratio=1)

        # encoder
        self.fc = nn.Linear(in_channels, hidden_channels)

        # hop_embedding
        self.hop_embedding = nn.Parameter(torch.randn(1, num_hop, hidden_channels))

        # interaction layers
        self.build_feature_inter_layer(feature_inter, hidden_channels, inter_layer)

        # fusion
        if self.feature_fusion == 'attention':
            self.atten_self = nn.Linear(hidden_channels, 1)
            self.atten_neighbor = nn.Linear(hidden_channels, 1)

        # prediction
        self.classifier = nn.Linear(hidden_channels, out_channels)

        # norm
        self.build_norm_layer(hidden_channels, inter_layer * 2 + 2)
        print('HopGNN hidden:', hidden_channels, 'interaction:', feature_inter, 'hop:', num_hop, 'layers:', inter_layer)

    def build_activation(self, activation):
        if activation == 'tanh':
            self.activate = torch.tanh
        elif activation == 'sigmoid':
            self.activate = torch.sigmoid
        elif activation == 'gelu':
            self.activate = F.gelu
        else:
            self.activate = F.relu

    def preprocess(self, adj, x):
        # 将 adj 移动到与 x 相同的设备上
        adj = adj.to(x.device)
        h0 = []
        for i in range(self.num_hop):
            h0.append(x)

            if adj.is_sparse:
                x = torch.sparse.mm(adj, x)
            else:
                x = x @ adj

            if i > 2: x = 0.7 * x
        self.h0 = torch.stack(h0, dim=1)
        self.pre = True
        return self.h0

    # def preprocess(self, adj, x):
    #     h0 = []
    #     for i in range(self.num_hop):
    #         h0.append(x)
    #         x = adj @ x
    #     self.h0 = torch.stack(h0, dim=1)
    #     self.pre = True
    #     return self.h0

    def build_feature_inter_layer(self, feature_inter, hidden_channels, inter_layer):
        self.interaction_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        if feature_inter == 'mlp':
            for i in range(inter_layer):
                mlp = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
                self.interaction_layers.append(mlp)
        elif feature_inter == 'gcn':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_GCN(hidden_channels))
        elif feature_inter == 'sage':
            for i in range(inter_layer):
                self.interaction_layers.append(Interaction_SAGE(hidden_channels))
        elif feature_inter == 'attention':
            for i in range(inter_layer):
                self.interaction_layers.append(
                    Interaction_Attention(hidden_channels, heads=4, dropout=0.1))
        else:
            self.interaction_layers.append(torch.nn.Identity())

    def build_norm_layer(self, hidden_channels, layers):
        self.norm_layers = nn.ModuleList()
        for i in range(layers):
            if self.norm_type == 'bn':
                self.norm_layers.append(nn.BatchNorm1d(self.num_hop))
            elif self.norm_type == 'ln':
                self.norm_layers.append(nn.LayerNorm(hidden_channels))
            else:
                self.norm_layers.append(nn.Identity())

    def norm(self, h, layer_index):
        h = self.norm_layers[layer_index](h)
        return h

    # N * hop * d => N * hop * d
    def embedding(self, h):
        h = self.dropout(h)
        h = self.fc(h)

        # h = self.acga(h)

        h = h + self.hop_embedding
        h = self.norm(h, 0)
        return h

    # N * hop * d => N * hop * d
    def interaction(self, h):
        inter_layers = len(self.interaction_layers)
        for i in range(inter_layers):
            h_prev = h
            h = self.dropout(h)
            h = self.interaction_layers[i](h)
            h = self.activate(h)
            h = h + h_prev
            # h = 0.5*h + 0.5*h_prev
            # alpha = torch.sigmoid(self.learnable_alpha)  # 可学习参数
            # h = alpha * h + (1 - alpha) * h_prev
            h = self.norm(h, i + 1)
        return h

    # N * hop * d =>  N * hop * d (concat) or N * d (mean/max/attention)
    def fusion(self, h):
        h = self.dropout(h)
        if self.feature_fusion == 'max':
            h = h.max(dim=1).values
        elif self.feature_fusion == 'attention':
            h_self, h_neighbor = h[:, 0, :], h[:, 1:, :]
            h_self_atten = self.atten_self(h_self).view(-1, 1)
            h_neighbor_atten = self.atten_neighbor(h_neighbor).squeeze()
            h_atten = torch.softmax(F.leaky_relu(h_self_atten + h_neighbor_atten), dim=1)
            h_neighbor = torch.einsum('nhd, nh -> nd', h_neighbor, h_atten).squeeze()
            h = h_self + h_neighbor
            # h = torch.sum(h * self.att_weights, dim=1)
        else:  # mean
            h = h.mean(dim=1)
        h = self.norm(h, -1)
        return h

    def build_hop(self, inputs):
        if len(inputs.shape) == 3:
            h = inputs
        else:
            if self.pre == False:
                self.h0 = self.preprocess(self.g, inputs)
            h = self.h0
        return h

    def forward(self, inputs, adj):
        h = self.agca(inputs, adj)

        h = self.build_hop(inputs)
        h = self.embedding(h)
        h = self.interaction(h)

        h = self.contra_norm(h)
        h = self.fusion(h)
        h = self.classifier(h)

        return h

    def forward_plus(self, inputs):
        h = self.build_hop(inputs)
        aug_h = torch.cat((h, h), dim=0)
        h = self.embedding(h)

        h = self.interaction(h)
        h = self.contra_norm(h)
        h = self.fusion(h)
        y = self.classifier(h)

        size = h.size(0)
        y1, y2 = y[:size // 2, ...], y[size // 2:, ...]
        view1, view2 = h[:size // 2, ...], h[size // 2:, ...]
        return (y1, y2), (view1, view2)
