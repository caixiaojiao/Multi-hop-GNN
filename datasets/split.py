# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
#
# def Stratified(datalist, labellist, size):
#     trainval_test = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
#     X_trainval = []
#     X_test = []
#     for train_index, test_index in trainval_test.split(datalist, labellist):
#         for i, j in enumerate(train_index):
#             X_trainval.append(datalist[j])
#         for i, j in enumerate(test_index):
#             X_test.append(datalist[j])
#     return X_trainval, X_test
#
# graph_list, label_list = load_data_from_mat_with_bit_HY(self.path)
# ### 训练集、验证集、测试集划分，6：2：2
# train_val_list, test_list = Stratified(graph_list, label_list, 0.2)
# train_val_label_list = []
# for data in train_val_list:
#     train_val_label_list.append((data.y).item())
# train_list, val_list = Stratified(train_val_list, train_val_label_list, 0.25)