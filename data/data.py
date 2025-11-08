# -*- coding: utf-8 -*-
"""
Created on 2024/6/17 7:52

@author: XUQIONG  (xuqiong@swpu.edu.cn)

"""

from torch.utils.data import Dataset, DataLoader, TensorDataset
from math import ceil
from utils import *
from data.show import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class Dataset_openfwi(Dataset):
    '''
       # Load the training data. It includes only seismic_data and vmodel.
       param_seismic_flag: for loading different frequency seismic data. It is designed for domain adaption.
       para_start_num: Start reading from the number of data.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        print("---------------------------------")
        print("Loading the datasets...")

        data_set, label_set = batch_read_npyfile(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)

        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        ###################################################################################
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_npyfile(para_dataset_dir,
                       para_start,
                       para_batch_length,
                       para_seismic_flag = "seismic",
                       para_train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
                                    edgeset    (500,1,70,70)   二值边缘图像
    '''



    dataset_list = []
    labelset_list = []

    for i in range(para_start, para_start + para_batch_length):

        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = para_dataset_dir + '{}_data/{}/seismic{}.npy'.format(para_train_or_test, para_seismic_flag, i)
        print("Reading: {}".format(filename_seis))

        data = np.load(filename_seis).astype(np.float32)
        dataset_list.append(data)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))
        label = np.load(filename_label).astype(np.float32)
        labelset_list.append(label)

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    return dataset, labelset


class Dataset_openfwi_test(Dataset):
    '''
       # Load the test data. It includes seismic_data, vmodel, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set = batch_read_npyfile(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)
        vmodel_max_min = np.zeros((para_train_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm)
            vmodel_max_min[i, 1] = np.min(vm)
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_openfwi4_test(Dataset):
    '''
       # Load the test data including velocity model edge.
       # It includes seismic_data, vmodel, edge, and vmodel_max_min.
       # vmodel_max_min is used for inverting normalization.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set, edge_set = batch_read_npyfile4(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)
        vmodel_max_min = np.zeros((para_train_size, 2))
        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            vmodel_max_min[i, 0] = np.max(vm)
            vmodel_max_min[i, 1] = np.min(vm)
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # ##################################################################################
        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).to(torch.uint8))
        self.vmodel_max_min = TensorDataset(torch.from_numpy(vmodel_max_min[:para_train_size, ...]).float())

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index], self.vmodel_max_min[index]

    def __len__(self):
        return len(self.seismic_data)


class Dataset_openfwi4(Dataset):
    '''
       Load the training data including velocity model edge. It includes seismic_data, vmodel, and edge of vmodel.
    '''

    def __init__(self, para_data_dir, para_train_size, para_start_num, para_seismic_flag, para_train_or_test):
        # para_start_num: 1 for train, 11 for test
        print("---------------------------------")
        print("· Loading the datasets...")

        data_set, label_set, edge_set = batch_read_npyfile4(para_data_dir, para_start_num, ceil(para_train_size / 500), para_seismic_flag, para_train_or_test)

        ###################################################################################
        #                          Vmodel normalization                                   #
        ###################################################################################
        print("Normalization in progress...")
        # for each sample
        for i in range(data_set.shape[0]):
            vm = label_set[i][0]
            label_set[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

        # Training set
        self.seismic_data = TensorDataset(torch.from_numpy(data_set[:para_train_size, ...]).float())
        self.vmodel = TensorDataset(torch.from_numpy(label_set[:para_train_size, ...]).float())
        self.edge = TensorDataset(torch.from_numpy(edge_set[:para_train_size, ...]).to(torch.uint8))

    def __getitem__(self, index):
        return self.seismic_data[index], self.vmodel[index], self.edge[index]

    def __len__(self):
        return len(self.seismic_data)


def batch_read_npyfile4(para_dataset_dir,
                       para_start,
                       para_batch_length,
                       para_seismic_flag = "seismic",
                       para_train_or_test = "train"):
    '''
    Batch read seismic gathers and velocity models for .npy file
    including velocity model edge.

    :param dataset_dir:             Path to the dataset
    :param start:                   Start reading from the number of data
    :param batch_length:            Starting from the defined first number of data, how long to read
    :param train_or_test:           Whether the read data is used for training or testing ("train" or "test")
    :return:                        a pair: (seismic data, velocity model, contour of velocity model)
                                    Among them, the dimensions of seismic data, velocity model and contour of velocity
                                    model are all (number of read data * 500, channel, height, width)
                                    dataset （500,5,1000,70）  地震数据  未经任何处理
                                    labelset, （500,1,70,70）  速度模型  未经任何处理
                                    edgeset    (500,1,70,70)   二值边缘图像
    '''

    dataset_list = []
    labelset_list = []
    edgeset_list = []

    for i in range(para_start, para_start + para_batch_length):
        ##############################
        ##    Load Seismic Data     ##
        ##############################

        # Determine the seismic data path in the dataset
        filename_seis = para_dataset_dir + '{}_data/{}/seismic{}.npy'.format(para_train_or_test, para_seismic_flag, i)
        print("Reading: {}".format(filename_seis))

        datas = np.load(filename_seis).astype(np.float32)
        dataset_list.append(datas)

        ##############################
        ##    Load Velocity Model   ##
        ##############################

        # Determine the velocity model path in the dataset
        filename_label = para_dataset_dir + '{}_data/vmodel/vmodel{}.npy'.format(para_train_or_test, i)
        print("Reading: {}".format(filename_label))
        labels = np.load(filename_label).astype(np.float32)
        labelset_list.append(labels)

        ###################################
        ##    Generating Velocity Edge   ##
        ###################################

        print("Generating velocity model profile......")
        edges = np.zeros([500, OutChannel, ModelDim[0], ModelDim[1]])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                edges[i, j, ...] = extract_contours(labels[i, j, ...])
        edgeset_list.append(edges.astype(np.uint8))

    dataset = np.concatenate(dataset_list, axis=0)
    labelset = np.concatenate(labelset_list, axis=0)
    edgeset = np.concatenate(edgeset_list, axis=0)

    return dataset, labelset, edgeset


if __name__ == '__main__':
    dataset_dir = 'D:/Xu-Qiong/Data/CurveFaultA/'
    # batch_read_npyfile(dataset_dir,1,2,train_or_test="train")
    # data_set = Dataset_openfwi_conv_train(dataset_dir, 1000, 1, "seismic", "train")
    # data_loader = DataLoader(data_set, batch_size=10, shuffle=True)
    #
    # # 遍历数据集
    # for (datas, labels) in data_loader:
    #     # 在这里，你可以使用批量的特征和目标进行训练或推断
    #
    #     print("ok ")

