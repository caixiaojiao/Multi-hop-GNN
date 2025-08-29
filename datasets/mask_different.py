# train_mask, val_mask, test_mask = generate_masks(label_np, seed=42)
# train_mask, val_mask, test_mask = generate_masks(num_of_nodes, label_np)

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def Stratified(datalist, labellist, size):
    trainval_test = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
    X_trainval = []
    X_test = []
    for train_index, test_index in trainval_test.split(datalist, labellist):
        for i, j in enumerate(train_index):
            X_trainval.append(datalist[j])
        for i, j in enumerate(test_index):
            X_test.append(datalist[j])
    return X_trainval, X_test

graph_list, label_list = load_data_from_mat_with_bit_HY(self.path)
### 训练集、验证集、测试集划分，6：2：2
train_val_list, test_list = Stratified(graph_list, label_list, 0.2)
train_val_label_list = []
for data in train_val_list:
    train_val_label_list.append((data.y).item())
train_list, val_list = Stratified(train_val_list, train_val_label_list, 0.25)



# 调用
# train_list, val_list, test_list = generate_masks(labels, node_features)

def generate_masks(class_labels, node_features):
    # 获取非零标签的索引
    valid_indices = np.where(class_labels >= 0)[0]
    valid_labels = class_labels[valid_indices]
    valid_node_features = node_features[valid_indices]

    def Stratified(features, labels, size):
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
        train_indices, test_indices = next(stratified_split.split(features, labels))
        train_list = [(features[i], labels[i]) for i in train_indices]
        test_list = [(features[i], labels[i]) for i in test_indices]
        return train_list, test_list

    # 首次分割得到训练+验证集 和 测试集
    train_val_list, test_list = Stratified(valid_node_features, valid_labels, 0.2)

    # 再次分割训练+验证集得到单独的训练集和验证集
    train_features, train_labels = zip(*train_val_list)
    train_list, val_list = Stratified(list(train_features), list(train_labels), 0.2)

    return train_list, val_list, test_list

# def generate_labels_and_masks(class_labels, train_ratio=0.7, val_test_ratio=0.3, seed=42):
#     # 获取非零标签的索引
#     valid_indices = np.where(class_labels >= 0)[0]
#     valid_labels = class_labels[valid_indices]
#
#     # 划分训练集和验证+测试集
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=val_test_ratio, random_state=seed)
#     train_indices, val_test_indices = next(sss.split(np.zeros(len(valid_indices)), valid_labels))
#
#     # 验证集和测试集使用相同的索引
#     val_indices = test_indices = val_test_indices
#
#     # 创建掩码
#     # train_mask = torch.zeros(class_labels.shape[0], dtype=torch.bool)
#     # val_mask = torch.zeros(class_labels.shape[0], dtype=torch.bool)
#     # test_mask = torch.zeros(class_labels.shape[0], dtype=torch.bool)
#     #
#     # train_mask[valid_indices[train_indices]] = True
#     # val_mask[valid_indices[val_indices]] = True
#     # test_mask[valid_indices[test_indices]] = True
#
#     # 转换索引为张量
#     train_mask = torch.tensor(valid_indices[train_indices], dtype=torch.int64)
#     val_mask = torch.tensor(valid_indices[val_indices], dtype=torch.int64)
#     test_mask = torch.tensor(valid_indices[test_indices], dtype=torch.int64)
#
#     return train_mask, val_mask, test_mask

# def generate_masks(num_of_nodes, labels, train_size=0.7, val_size=0.15, test_size=0.15):
#     assert train_size + val_size + test_size == 1.0, \
#         "Train, validation, and test sizes must sum to 1."
#
#     # Initialize masks as all False
#     train_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
#     val_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
#     test_mask = torch.zeros(num_of_nodes, dtype=torch.bool)
#
#     # Split into train+val and test sets
#     trainval_test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
#     trainval_idx, test_idx = next(trainval_test_split.split(np.zeros(num_of_nodes), labels))
#
#     # Further split train+val into train and val sets
#     train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (train_size + val_size), random_state=42)
#     train_idx, val_idx = next(train_val_split.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
#
#     # Assign the masks
#     train_mask[trainval_idx[train_idx]] = True
#     val_mask[trainval_idx[val_idx]] = True
#     test_mask[test_idx] = True
#
#     return train_mask, val_mask, test_mask

# def generate_masks(total_nodes, class_labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
#     # 筛选出类别不为0的节点
#     valid_indices = np.where(class_labels != 0)[0]
#     valid_labels = class_labels[valid_indices]
#
#     # 第一次分割：训练集和验证+测试集
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=val_ratio + test_ratio, random_state=seed)
#     train_indices, val_test_indices = next(sss.split(np.zeros(len(valid_indices)), valid_labels))
#
#     # 第二次分割：验证集和测试集
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio / (val_ratio + test_ratio),
#                                  test_size=1 - (val_ratio / (val_ratio + test_ratio)), random_state=seed)
#     val_indices, test_indices = next(sss.split(np.zeros(len(val_test_indices)), valid_labels[val_test_indices]))
#
#     # 创建掩码
#     # train_mask = torch.zeros(total_nodes, dtype=torch.bool)
#     # val_mask = torch.zeros(total_nodes, dtype=torch.bool)
#     # test_mask = torch.zeros(total_nodes, dtype=torch.bool)
#     #
#     # train_mask[valid_indices[train_indices]] = True
#     # val_mask[valid_indices[val_test_indices[val_indices]]] = True
#     # test_mask[valid_indices[val_test_indices[test_indices]]] = True
#
#     train_mask = torch.tensor(train_indices, dtype=torch.int64)
#     val_mask = torch.tensor(val_test_indices[val_indices], dtype=torch.int64)
#     test_mask = torch.tensor(val_test_indices[test_indices], dtype=torch.int64)
#
#
#     return train_mask, val_mask, test_mask

# def generate_masks(total_nodes, class_labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
#     # 筛选类别为1到5的节点
#     valid_indices = np.where(class_labels != 0)[0]
#     valid_labels = class_labels[valid_indices]
#
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=val_ratio + test_ratio,
#                                  random_state=seed)
#     train_indices, val_test_indices = next(sss.split(np.zeros(len(valid_indices)), valid_labels))
#
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio / (val_ratio + test_ratio),
#                                  test_size=1 - (val_ratio / (val_ratio + test_ratio)), random_state=seed)
#     val_indices, test_indices = next(sss.split(np.zeros(len(val_test_indices)), valid_labels[val_test_indices]))
#
#     train_mask = torch.tensor(valid_indices[train_indices], dtype=torch.int64)
#     val_mask = torch.tensor(valid_indices[val_test_indices[val_indices]], dtype=torch.int64)
#     test_mask = torch.tensor(valid_indices[val_test_indices[test_indices]], dtype=torch.int64)
#
#     return train_mask, val_mask, test_mask
#
# new!
# def generate_masks(class_labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
#     unique_classes = np.unique(class_labels)
#     train_indices, val_indices, test_indices = [], [], []
#
#     for cls in unique_classes:
#         indices = np.where(class_labels == cls)[0]
#         num_samples = len(indices)
#
#         # 确定每个集合的数量
#         num_train = int(num_samples * train_ratio)
#         num_val = int(num_samples * val_ratio)
#         num_test = num_samples - num_train - num_val  # 保证总数正确
#
#         # 使用StratifiedShuffleSplit进行分割
#         sss = StratifiedShuffleSplit(n_splits=1, train_size=num_train, test_size=num_val + num_test, random_state=seed)
#         train_idx, val_test_idx = next(sss.split(np.zeros(num_samples), class_labels[indices]))
#
#         sss = StratifiedShuffleSplit(n_splits=1, train_size=num_val, test_size=num_test, random_state=seed)
#         val_idx, test_idx = next(sss.split(np.zeros(len(val_test_idx)), class_labels[indices[val_test_idx]]))
#
#         # 将当前类别的索引按分割结果添加到相应列表中
#         train_indices.extend(indices[train_idx])
#         val_indices.extend(indices[val_test_idx][val_idx])
#         test_indices.extend(indices[val_test_idx][test_idx])
#
#     # 将索引列表转换为PyTorch张量
#     train_mask = torch.tensor(train_indices, dtype=torch.int64)
#     val_mask = torch.tensor(val_indices, dtype=torch.int64)
#     test_mask = torch.tensor(test_indices, dtype=torch.int64)
#
#     return train_mask, val_mask, test_mask
#
# 训练集和验证集相同，剩下用来预测
# def generate_masks(class_labels, num_samples_per_class=50, seed=None):
#     np.random.seed(seed)  # 设置随机种子以确保可重复性
#     unique_classes = np.unique(class_labels)
#     train_indices, val_indices, test_indices = [], [], []
#
#     for cls in unique_classes:
#         indices = np.where(class_labels == cls)[0]
#         np.random.shuffle(indices)  # 随机打乱索引
#
#         # 选择每个类别的50个样本作为train和val，剩余的作为test
#         train_val_indices = indices[:num_samples_per_class]
#         test_indices.extend(indices[num_samples_per_class:])
#
#         # 因为train_mask和val_mask相同，所以直接复制
#         train_indices.extend(train_val_indices)
#         val_indices.extend(train_val_indices)
#
#     train_mask = torch.tensor(train_indices, dtype=torch.int64)
#     val_mask = torch.tensor(val_indices, dtype=torch.int64)
#     test_mask = torch.tensor(test_indices, dtype=torch.int64)
#
#     return train_mask, val_mask, test_mask
#
# 遍历各个类别按0.6,0.2,0.2
# def generate_masks(class_labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
#     unique_classes = np.unique(class_labels)
#     train_indices, val_indices, test_indices = [], [], []
#
#     for cls in unique_classes:
#         indices = np.where(class_labels == cls)[0]
#         num_samples = len(indices)
#
#         num_train = int(num_samples * train_ratio)
#         num_val = int(num_samples * val_ratio)
#         num_test = num_samples - num_train - num_val
#
#         sss = StratifiedShuffleSplit(n_splits=1, train_size=num_train, test_size=num_val + num_test, random_state=seed)
#         train_idx, val_test_idx = next(sss.split(np.zeros(num_samples), class_labels[indices]))
#
#         sss = StratifiedShuffleSplit(n_splits=1, train_size=num_val, test_size=num_test, random_state=seed)
#         val_idx, test_idx = next(sss.split(np.zeros(len(val_test_idx)), class_labels[indices[val_test_idx]]))
#
#         train_indices.extend(indices[train_idx])
#         val_indices.extend(indices[val_test_idx][val_idx])
#         test_indices.extend(indices[val_test_idx][test_idx])
#
#     train_mask = torch.tensor(train_indices, dtype=torch.int64)
#     val_mask = torch.tensor(val_indices, dtype=torch.int64)
#     test_mask = torch.tensor(test_indices, dtype=torch.int64)
#
#     return train_mask, val_mask, test_mask
#
# 平衡类别的分层随机采样 精度img1_graph = 0.9035
# def generate_masks(total_nodes, class_labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, test_size=val_ratio + test_ratio,
#                                  random_state=seed)
#     train_indices, val_test_indices = next(sss.split(np.zeros(total_nodes), class_labels))
#
#     sss = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio / (val_ratio + test_ratio),
#                                  test_size=1 - (val_ratio / (val_ratio + test_ratio)), random_state=seed)
#     val_indices, test_indices = next(sss.split(np.zeros(len(val_test_indices)), class_labels[val_test_indices]))
#
#     train_mask = torch.tensor(train_indices, dtype=torch.int64)
#     val_mask = torch.tensor(val_test_indices[val_indices], dtype=torch.int64)
#     test_mask = torch.tensor(val_test_indices[test_indices], dtype=torch.int64)
#
#     return train_mask, val_mask, test_mask
#
#
# 分层随机采样（未指定类别的样本数）
# def generate_masks(total_nodes, class_labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
#
#     if seed is not None:
#         np.random.seed(seed)
#     indices = np.arange(total_nodes)
#     np.random.shuffle(indices)
#
#     num_train = int(total_nodes * train_ratio)
#     num_val = int(total_nodes * val_ratio)
#     num_test = total_nodes - num_train - num_val
#
#     unique_labels = np.unique(class_labels)
#     num_classes = len(unique_labels)
#
#     train_counts = {label: 0 for label in unique_labels}
#     val_counts = {label: 0 for label in unique_labels}
#     test_counts = {label: 0 for label in unique_labels}
#
#     train_indices = []
#     val_indices = []
#     test_indices = []
#
#     for idx in indices:
#         label = class_labels[idx]
#
#         if train_counts[label] < num_train / num_classes:
#             train_indices.append(idx)
#             train_counts[label] += 1
#         elif val_counts[label] < num_val / num_classes:
#             val_indices.append(idx)
#             val_counts[label] += 1
#         else:
#             test_indices.append(idx)
#             test_counts[label] += 1
#
#     train_mask = torch.as_tensor(train_indices)
#     val_mask = torch.as_tensor(val_indices)
#     test_mask = torch.as_tensor(test_indices)
#
#     return train_mask, val_mask, test_mask