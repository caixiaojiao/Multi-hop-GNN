import scipy.io as sio
import torch
import numpy as np
from pymatreader import read_mat
from scipy.sparse import csc_matrix
from torch_sparse import SparseTensor
from datasets.graph_transform import sparse_normalize, remove_self_loop
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit


class GraphData:
    def __init__(self, x, edge_index, y, adj, num_of_nodes, num_of_class, train_mask, val_mask, test_mask):

        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.adj = adj
        self.num_of_nodes = num_of_nodes
        self.num_of_class = num_of_class
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

def split_indices(labels, seed=42):
    idx = list(range(len(labels)))

    train_idx, test_val_idx = train_test_split(idx, test_size=0.3, random_state=seed, stratify=labels)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=seed, stratify=[labels[i] for i in test_val_idx])

    return train_idx, val_idx, test_idx


# D-1/2 * A * D-1/2 or D-1 * A
def sparse_normalize(adj, symmetric_norm=True):
    assert isinstance(adj, SparseTensor)
    size = adj.size(0)
    ones = torch.ones(size).view(-1, 1).to(adj.device())
    degree = adj @ ones
    if symmetric_norm == False:
        degree = degree ** -1
        degree[torch.isinf(degree)] = 0
        return adj * degree
    else:
        degree = degree ** (-1 / 2)
        degree[torch.isinf(degree)] = 0
        d = SparseTensor(row=torch.arange(size).to(adj.device()), col=torch.arange(size).to(adj.device()),
                         value=degree.squeeze().to(adj.device()),
                         sparse_sizes=(size, size)).to(adj.device())
        adj = adj @ d
        adj = adj * degree
        return adj


def Init_matrix(edge_index):
    edge_index = remove_self_loop(edge_index)
    adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :],
                       sparse_sizes=(torch.max(edge_index) + 1, torch.max(edge_index) + 1))
    adj = adj.cuda()
    sym_norm = True
    adj = sparse_normalize(adj, sym_norm).cuda()

    return adj


def load_data_from_mat(path, device):
    mat_data = read_mat(path)
    edge_index = torch.from_numpy(mat_data['edg'].astype(np.int64)).t().contiguous() - 1
    node_features = torch.from_numpy(mat_data['total_multi_feature']).float()
    labels = torch.from_numpy(mat_data['reg_cls'].astype(np.int64)) - 1

    adj = Init_matrix(edge_index)

    x = node_features.to(device)
    num_of_nodes = x.shape[0]
    num_of_class = 5

    dataset_idx = torch.where(labels != -1)[0]
    dataset_label = labels[dataset_idx].numpy()

    train_mask, val_mask, test_mask = split_indices(dataset_label, seed=42)

    graph = GraphData(x=x, edge_index=edge_index, y=labels, adj=adj, num_of_nodes=num_of_nodes,
                      num_of_class=num_of_class, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return graph







