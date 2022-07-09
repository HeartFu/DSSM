"""
DSSM 双塔模型
"""

import sys

import torch.nn as nn

from utils.utils import get_activation

sys.path.append('../')
from model.EmbeddingModule import EmbeddingModule


class DSSM(nn.Module):
    def __init__(self, user_datatypes, item_datatypes, user_dnn_size=(256, 128),
                 item_dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False):
        super().__init__()
        self.user_dnn_size = user_dnn_size
        self.item_dnn_size = item_dnn_size
        self.dropout = dropout
        self.user_datatypes = user_datatypes
        self.item_datatypes = item_datatypes

        # 用户侧
        self.user_tower = Tower(self.user_datatypes, self.user_dnn_size, self.dropout, activation=activation, use_senet=use_senet)
        self.item_tower = Tower(self.item_datatypes, self.item_dnn_size, self.dropout, activation=activation, use_senet=use_senet)

    def forward(self, user_feat, item_feat):
        return self.user_tower(user_feat), self.item_tower(item_feat)


class Tower(nn.Module):
    def __init__(self, datatypes, dnn_size=(256, 128), dropout=0.0, activation='ReLU', use_senet=False):
        super().__init__()
        self.dnns = nn.ModuleList()
        self.embeddings = EmbeddingModule(datatypes, use_senet)
        input_dims = self.embeddings.sparse_dim + self.embeddings.dense_dim
        # self.use_senet = use_senet
        # if self.use_senet:
        #     self.se_net = SENet(input_dims)
        for dim in dnn_size:
            self.dnns.append(nn.Linear(input_dims, dim))
            # self.dnns.append(nn.BatchNorm1d(dim))
            self.dnns.append(nn.Dropout(dropout))
            self.dnns.append(get_activation(activation))
            input_dims = dim

    def forward(self, x):
        dnn_input = self.embeddings(x)
        # print(dnn_input.type())
        # if self.use_senet:
        #     dnn_input = self.se_net(dnn_input)

        # print(torch.mean(self.dnns[0].weight))
        for dnn in self.dnns:
            # if self.training == False:
            #     import pdb
            #     pdb.set_trace()
            dnn_input = dnn(dnn_input)

        # print('finish!')
        return dnn_input
