# coding=utf-8
"""
DSSM Dataset
"""
from sklearn.utils import shuffle
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from utils.utils import get_datatypes, data_to_type_list


class DSSMDataSet(Dataset):
    def __init__(self, data, y, cols_param=None, datatype=None,
                 label_encoder_rate_min=0.001, excepts=None, training=False,
                 neg_sampling=0):
        # 先获取datatype
        self.cols_param = cols_param
        if datatype is None:
            self.datatypes = get_datatypes(data, cols_param,
                                           label_encoder_rate_min=label_encoder_rate_min,
                                           excepts=excepts)
        else:
            self.datatypes = datatype
        # format数据
        print('Start to format data! Please wait a moment! It will be taken a long time!')

        self.y = y.astype(np.float)

        if training and neg_sampling > 0:
            data = self.negative_sampling(data, neg_sampling)
        print(data.head())
        # print(pd.DataFrame(self.datatypes))
        features = data_to_type_list(data, self.datatypes)
        self.user_feat = features['user']
        self.item_feat = features['item']
        print(self.item_feat[0])
        print(self.user_feat[0])
        # print(data.iloc[0])
        # print(self.user_feat[0], self.item_feat[0])


    def __getitem__(self, index):
        return self.user_feat[index], self.item_feat[index], self.y[index]

    def __len__(self):
        return len(self.y)

    def negative_sampling(self, data, sampling_cnt):
        print('Start to get negative_sampling! len of data is {}'.format(len(data)))
        user_cols = []
        item_cols = []
        for key in self.cols_param.keys():
            if self.cols_param[key]['belongs'] == 'user':
                user_cols.append(key)
            else:
                item_cols.append(key)

        user_data = data[user_cols]
        item_data = data[item_cols]

        print('Enter to get negative item data!')
        neg_data_item = item_data.sample(sampling_cnt * len(item_data), replace=True, random_state=None, axis=0)
        print('Enter to get negative user data!')
        # neg_data_item = pd.concat(neg_list, axis=0)
        neg_data_user = pd.concat([user_data for _ in range(sampling_cnt)], axis=0)
        print('Enter to shuffle data')
        neg_data_item = shuffle(neg_data_item)
        neg_data_user = shuffle(neg_data_user)

        print('Enter to concat user and item data')
        neg_data = pd.concat([neg_data_user.reset_index(drop=True), neg_data_item.reset_index(drop=True)], axis=1)

        all_data = pd.concat([data, neg_data], axis=0)

        self.y = np.concatenate((self.y, np.asarray([[0.] for _ in range(len(neg_data))])), axis=0)

        return all_data