import os
import sys
import scipy.io as sio
from scipy.io import savemat

sys.path.append('..')
from models.multihopgnn import HopGNN
from datasets.dataloader import *
from utils.utils import *
import argparse
# CUDA_LAUNCH_BLOCKING=1
import torch.nn.functional as F
from torch_geometric.data import Data
import torch, gc

gc.collect()
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
# Training hyper-parameters
parser.add_argument('--seed', type=int, default=202, help='set Random seed.')
parser.add_argument('--cuda_id', type=str, default='0', help='CUDA id')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=40000, help="batch size for the mini-batch")
parser.add_argument('--log_dur', type=int, default=50, help='interval of epochs for log during training.')
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='weight decay (L2 loss on parameters).')
parser.add_argument('--ssl', type=bool, default=0, help="use hopgnn+ssl or not")
parser.add_argument('--alpha', type=float, default=0.5, help="the alpha of de-correlation in Barlow Twins")
parser.add_argument('--lambd', type=float, default=5e-4, help="the weight of SSL objective")

# Model hyper-parameters
parser.add_argument("--graph_path", default= r'E:\new\Subset2\Mat\Subset2_graph_multi_233.mat', type=str, help="path to img mat files")
parser.add_argument('--save_dir', type=str, default=r'F:\Model_run\HopGNN-CLC\pth_best', help='directory to save models')
parser.add_argument('--predictions', type=str, default=r'F:\Model_run\HopGNN-CLC\result', help='predictions of models')
parser.add_argument('--model', type=str, default='hopgnn', help='which models')
parser.add_argument('--hidden', type=int, default=128, help='number of hidden units.')
parser.add_argument('--num_layer', type=int, default=3, help='number of interaction layer')
parser.add_argument('--num_hop', type=int, default=6, help='number of hop information')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (1 - keep probability).')
parser.add_argument('--interaction', type=str, default='sage', help='feature interaction type of HopGNN')
parser.add_argument('--fusion', type=str, default='max', help='feature fusion type of HopGNN')
parser.add_argument('--activation', type=str, default='relu', help="activation function")
parser.add_argument('--norm_type', type=str, default='ln', help="the normalization type")
args = parser.parse_args()


def get_loss(model, features, labels, args):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # HopGNN+ SSL
    if args.ssl == 1 and isinstance(model, HopGNN):
        ssl_func = BarlowLoss(args.alpha)
        (y1, y2), (view1, view2) = model.forward_plus(features)
        ce_loss = loss_func(y1, labels.long()) + loss_func(y2, labels.long()) + 1e-9
        ssl_loss = ssl_func(view1, view2)
        output = y1
        loss = ce_loss + args.lambd * ssl_loss
    else:
        output = model(features)
        loss = loss_func(output, labels.long()) + 1e-9
    return output, loss

# mini-batch training
def roll_an_epoch(model, features, labels, mask, optimizer, batch_size, manner, args):
    # if manner == 'train':
    #     # shuffle the train mask for mini-batch training
    #     mask = mask[torch.randperm(len(mask))]
    # Convert boolean mask to index tensor if necessary
    if isinstance(mask, list):
        mask = torch.tensor(mask)
    if mask.dtype == torch.bool:
        mask = torch.where(mask)[0]
    # shuffle the train mask for mini-batch training
    mask = mask[torch.randperm(len(mask))]
    device = features.device
    total_loss = []
    total_output = []
    total_label = []

    for i in range(0, len(mask), batch_size):
        # generate batch index , features, label
        index = mask[i:i + batch_size]
        batch_features = features[index].to(device)
        batch_label = labels[index].to(device)
        if manner == 'train':
            model.train()
            batch_output, loss = get_loss(model, batch_features, batch_label, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                batch_output, loss = get_loss(model, batch_features, batch_label, args)

        total_loss.append(loss.cpu().item())
        total_output.append(batch_output)
        total_label.append(batch_label)

    loss = np.mean(total_loss)
    total_output = torch.cat(total_output, dim=0)
    total_label = torch.cat(total_label)
    acc = get_accuracy(total_output, total_label)
    log_info = {'loss': loss, 'acc': acc}
    return log_info

def train(epoch, model, features, labels, train_mask, val_mask, test_mask, args):
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    set_seed(202)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], [], []
    best_val_acc, best_test_acc = 0, 0
    best_epoch = 0
    for iter in range(epoch):
        train_log_info = roll_an_epoch(model, features, labels, train_mask, optimizer, batch_size=batch_size, manner='train', args=args)
        val_log_info = roll_an_epoch(model, features, labels, val_mask, optimizer, batch_size=batch_size, manner='val', args=args)
        test_log_info = roll_an_epoch(model, features, labels, test_mask, optimizer, batch_size=batch_size, manner='test', args=args)

        #log info
        train_loss_list.append(train_log_info['loss'])
        val_loss_list.append(val_log_info['loss'])
        train_acc, val_acc, test_acc = train_log_info['acc'], val_log_info['acc'], test_log_info['acc']
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        #update best test via val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = iter

        if (iter + 1) % args.log_dur == 0:
            print(
                "Epoch {:4d}, Train_loss {:.4f}, Val_loss {:.4f}, train_acc {:.4f},  val_acc {:.4f}, test_acc{:.4f}".format(
                    iter + 1, np.mean(train_loss_list), np.mean(val_loss_list), train_acc, val_acc, test_acc))

    best_model_path = fr'F:\Model_run\HopGNN-CLC\pth_best\{best_epoch}.pth'
    torch.save(model.state_dict(), best_model_path)

    print("Best at {} epoch, Val Accuracy {:.4f} Test Accuracy {:.4f}".format(best_epoch, best_val_acc, best_test_acc))
    return best_test_acc, best_model_path

def test(model, features, label):
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        acc = get_accuracy(outputs, label)
    return acc


def Init_matrix(edge_index):
    edge_index = remove_self_loop(edge_index)
    adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :],
                       sparse_sizes=(torch.max(edge_index) + 1, torch.max(edge_index) + 1))
    # adj = SparseTensor(row=edge_index[0, :], col=edge_index[1, :],
    #                    sparse_sizes=(self.num_of_nodes, self.num_of_nodes))
    # adj = adj.to(self.device)
    adj = adj.cuda()
    # if 'product' in self.name:
    #     sym_norm = False
    # else:
    sym_norm = True
    # self.adj = sparse_normalize(adj, sym_norm).to(self.device)
    adj = sparse_normalize(adj, sym_norm).cuda()

    return adj


def load_data_big(path, device):
    mat_data = read_mat(path)
    edge_indices = torch.from_numpy(mat_data['edg'].astype(np.int64)).t().contiguous() - 1
    node_features = torch.from_numpy(mat_data['total_multi_feature']).float()
    labels = torch.from_numpy(mat_data['reg_cls'].astype(np.int64)) - 1
    graph = Data(x=node_features.to(device), edge_index=edge_indices.to(device), y=labels.to(device))
    return graph


def merge_graphs(graph_list):

    edge_indices = []
    num_nodes = 0

    node_features = []
    labels = []

    for graph in graph_list:

        edge_index = graph.edge_index + num_nodes
        edge_indices.append(edge_index)

        node_features.append(graph.x)
        labels.append(graph.y)

        num_nodes += graph.num_nodes

    block_diag_x = torch.cat(node_features)

    labels = torch.cat(labels)

    block_diag_edge_index = torch.cat(edge_indices, dim=1)

    return Data(x=block_diag_x, edge_index=block_diag_edge_index, y=labels)


def merge_adj_matrices(adj_list):

    adj_matrices = [adj.to_dense() for adj in adj_list]
    block_diag_adj = torch.block_diag(*adj_matrices)

    return block_diag_adj


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    directory_path = r'F:\Exp1\Subset_cat_multi_small'
    files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.mat')])

    graph_list = []
    adj_list = []

    for file in files:
        graph = load_data_big(file, device)
        adj = Init_matrix(graph.edge_index)
        graph_list.append(graph)
        adj_list.append(adj)

    merged_graph = merge_graphs(graph_list)
    merged_adj = merge_adj_matrices(adj_list)

    in_dim, hid_dim, out_dim = merged_graph.x.shape[1], args.hidden, 5
    model = HopGNN(merged_adj, in_dim, hid_dim, out_dim, args.num_hop, args.dropout, feature_inter=args.interaction, activation=args.activation,
                   inter_layer=args.num_layer, feature_fusion=args.fusion, norm_type=args.norm_type).to(device)

    merged_graph.x = model.preprocess(merged_adj, merged_graph.x)
    dataset_idx = torch.where(merged_graph.y != -1)[0]
    dataset_label = merged_graph.y.cpu().numpy()[dataset_idx.cpu().numpy()]
    train_idx, test_idx, val_idx = split_indices(dataset_label)

    best_epoch, best_model_path = train(epoch=args.epochs, model=model, features=merged_graph.x, labels=merged_graph.y,
                       train_mask=train_idx, val_mask=val_idx, test_mask=test_idx, args=args)
