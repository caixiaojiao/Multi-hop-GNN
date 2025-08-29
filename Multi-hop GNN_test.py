# -*- coding: utf-8 -*-
# @Time : 2024/9/2 20:05
# @Author : Rocco
import math
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pymatreader import read_mat
from scipy.io import savemat,loadmat
from scipy.sparse import csc_matrix
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from torch import flatten
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
from collections import Counter
from datasets.graph_transform import sparse_normalize, remove_self_loop
from models.multihopgnn import HopGNN
from utils import set_seed, BarlowLoss
from skimage import io, measure
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.sparse import csc_matrix

class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(target)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    def _class_weights(self, target):
        # Calculate class frequencies based on the target
        class_counts = torch.bincount(target[target != self.ignore_index], minlength=5).float()
        total_count = class_counts.sum()

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-5
        class_weights = total_count / (class_counts + epsilon)

        return class_weights.to(target.device)

def split_indices(labels, seed=42):
    idx = list(range(len(labels)))

    train_idx, test_val_idx = train_test_split(idx, test_size=0.9, random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.1, random_state=seed, stratify=[labels[i] for i in test_val_idx])

    return train_idx, val_idx, test_idx

def evaluate(fea_, label_):

    if isinstance(label_, np.ndarray):
        label_ = torch.from_numpy(label_).to(fea_.device)

    _, preds = torch.max(fea_, 1)
    preds_np = preds.detach().cpu().numpy()
    labels_np = label_.detach().cpu().numpy()
    oa = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='weighted')

    return oa, f1


def Init_matrix(edge_index, node_coords, osm_file_path):

    osm_image = io.imread(osm_file_path)
    labeled_image = measure.label(osm_image, connectivity=2)
    regions = measure.regionprops(labeled_image)

    node_coords_np = node_coords.cpu().numpy()
    node_regions = labeled_image[node_coords_np[:, 0], node_coords_np[:, 1]]

    region_mask = np.zeros((len(node_regions), len(node_regions)), dtype=bool)
    for i in range(len(node_regions)):
        region_mask[i] = (node_regions == node_regions[i])

    edge_np = edge_index.cpu().numpy()

    mask = region_mask[edge_np[0], edge_np[1]]
    filtered_edges = edge_np[:, mask]

    filtered_edge_index = torch.from_numpy(filtered_edges).long().contiguous()
    filtered_edge_index = remove_self_loop(filtered_edge_index)

    adj = SparseTensor(row=filtered_edge_index[0, :],
                       col=filtered_edge_index[1, :],
                       sparse_sizes=(torch.max(filtered_edge_index) + 1,
                                     torch.max(filtered_edge_index) + 1))
    adj = adj.cuda()
    adj = sparse_normalize(adj, True)  # 对称归一化
    return adj


def load_sparse_matrix(mat_group, num_nodes, device):

    data = np.array(mat_group['data'], dtype=np.float32)
    ir = np.array(mat_group['ir'], dtype=np.int32)
    jc = np.array(mat_group['jc'], dtype=np.int32)

    sparse_mat = csc_matrix((data, ir, jc), shape=(num_nodes, num_nodes))

    dense_mat = torch.from_numpy(sparse_mat.todense()).float().to(device)
    return dense_mat


def load_data_big(path, device):

    mat_data = h5py.File(path, 'r')

    edge_indices = torch.from_numpy(np.array(mat_data['edg'], dtype=np.int64)).t().contiguous() - 1

    node_features = torch.from_numpy(np.array(mat_data['total_multi_feature'], dtype=np.float32)).float()

    labels = torch.from_numpy(np.array(mat_data['reg_cls'], dtype=np.int64)) - 1

    num_nodes = mat_data['total_multi_feature'].shape[1]
    adj_matrix_1 = load_sparse_matrix(mat_data['adj'], num_nodes, device)
    adj_matrix_2 = load_sparse_matrix(mat_data['adj_osm'], num_nodes, device)
    adj_matrix_3 = load_sparse_matrix(mat_data['adj_weighted'], num_nodes, device)

    graph = Data(x=node_features.to(device), edge_index=edge_indices.to(device), y=labels.to(device))

    return graph, adj_matrix_3


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat_file_big = '.\datasets\mat\Graph.mat'
    num_classes = 5
    epochs = 2500
    set_seed(202)
    graph_big, adj_big = load_data_big(mat_file_big,device)

    dataset_idx = torch.where(graph_big.y != -1)[0]
    dataset_label = graph_big.y.cpu().numpy()[dataset_idx.cpu().numpy()]
    train_idx, test_idx, val_idx = split_indices(dataset_label)

    feature_inter = 'gcn'
    activation = 'relu'
    inter_layer = 2
    num_hop = 5
    feature_fusion = 'max'
    model = HopGNN(adj_big, in_channels=233, hidden_channels=128,
               out_channels=5, num_hop=num_hop, dropout=0.3, feature_inter=feature_inter,
               activation=activation, inter_layer=inter_layer, feature_fusion=feature_fusion, norm_type='ln').cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    train_loss, val_loss, test_loss = [], [], []
    train_f1_scores, val_f1_scores, test_f1_scores = [], [], []
    best_epoch, best_train_oa, best_train_f1, best_val_oa, best_val_f1 = -1, 0, 0, 0, 0

    patience = 50
    min_delta = 0.001
    counter = 0
    patton = 'train'
    if patton == 'train':
        for epoch in range(epochs):
            model.train()
            # output = model(hop_x, graph_big.edge_index)
            # output = model(graph_big.x)
            output = model(graph_big.x, adj_big)

            # ssl_func = BarlowLoss(0.5)
            # (y1, y2), (view1, view2) = model(hop_x)
            # ce_loss = criterion(y1, graph.y.long()) + criterion(y2, graph.y.long()) + 1e-9
            # ce_loss = criterion(y1.squeeze(), graph_big.y.long().squeeze()) + \
            #           criterion(y2.squeeze(), graph_big.y.long().squeeze()) + 1e-9
            # ssl_loss = ssl_func(view1, view2)
            # output = y1

            fea_ = output[graph_big.y != -1]
            label_ = graph_big.y[graph_big.y != -1]

            fea_train = fea_[train_idx]
            fea_valid = fea_[val_idx]
            fea_test = fea_[test_idx]

            label_train = label_[train_idx]
            label_valid = label_[val_idx]
            label_test = label_[test_idx]

            loss_train = criterion(fea_train, label_train.long())
            loss_valid = criterion(fea_valid, label_valid.long())
            loss_test = criterion(fea_test, label_test.long())

            oa_train, f1_train = evaluate(fea_train, label_train)
            oa_valid, f1_valid = evaluate(fea_valid, label_valid)
            oa_test, f1_test = evaluate(fea_test, label_test)

            optimizer.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 控制梯度爆炸
            optimizer.step()
            # 更新学习率
            scheduler.step(loss_valid)  # 根据验证损失调整

            # Only print every 25 epochs to avoid duplication
            if epoch % 50 == 0:
                train_loss.append(loss_train.item())
                val_loss.append(loss_valid.item())
                test_loss.append(loss_test.item())

                train_f1_scores.append(f1_train)
                val_f1_scores.append(f1_valid)
                test_f1_scores.append(f1_test)

                print(f"Epoch:{epoch} | Train Loss: {loss_train:.4f} | Val Loss: {loss_valid:.4f} | Test Loss: {loss_test:.4f}")
                print(f"Train F1: {f1_train:.4f} | Val F1: {f1_valid:.4f} | Test F1: {f1_test:.4f}")

            # Keep track of best performance
            if oa_train > best_val_oa:
                best_epoch = epoch
                best_train_oa = oa_train
                best_train_f1 = f1_train
                best_val_oa = oa_valid
                best_val_f1 = f1_valid
                best_predictions = torch.max(output.detach().cpu(), 1)[1].numpy().reshape(-1, 1)
            best_val_loss = float('inf')
            if loss_valid < best_val_loss - min_delta:
                best_val_loss = loss_valid
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        print(f'Best train_OA Epoch: {best_epoch}')
        print(f'Best Training OA: {best_train_oa:.4f}, Best Training F1: {best_train_f1:.4f}')
        print(f'Best Validation OA: {best_val_oa:.4f}, Best Validation F1: {best_val_f1:.4f}')

    else:
        def load_data_test(path, device):
            mat_data = read_mat(path)
            edge_indices = torch.from_numpy(mat_data['edg'].astype(np.int64)).t().contiguous() - 1
            # node_features = torch.from_numpy(mat_data['total_multi_feature']).float()
            node_features = torch.from_numpy(mat_data['total_binary_feature']).float()
            labels = torch.from_numpy(mat_data['reg_bd'].astype(np.int64)) - 1
            # labels = torch.from_numpy(mat_data['reg_cls'].astype(np.int64)) - 1
            graph = Data(x=node_features.to(device), edge_index=edge_indices.to(device), y=labels.to(device))
            return graph





