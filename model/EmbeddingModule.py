"""
Embedding Module 类
包含4中Embedding
"""

import torch
import torch.nn as nn

from model.SENet import SENet


class EmbeddingModule(nn.Module):
    def __init__(self, datatypes, use_se_net):
        super().__init__()
        self.dense_dim = 0
        self.sparse_dim = 0
        self.embs = nn.ModuleList()
        self.datatypes = datatypes
        self.use_se_net = use_se_net
        self.sparse_num = 0
        self.dense_num = 0
        # self.cols = cols
        for datatype in datatypes:
            # datatype = datatypes[col]
            if datatype['type'] == 'SparseEncoder' or datatype['type'] == 'BucketSparseEncoder':
                #                 if datatype['length'] > 15:
                self.embs.append(nn.Embedding(datatype['length'], datatype['emb_dim']))
                self.sparse_dim += datatype['emb_dim']
                self.sparse_num += 1
            if datatype['type'] == 'MultiSparseEncoder':
                self.embs.append(nn.EmbeddingBag(datatype['length'], datatype['emb_dim'], mode='sum'))
                self.sparse_dim += datatype['emb_dim']
                self.sparse_num += 1
            elif datatype['type'] == 'DenseEncoder':
                self.dense_dim += 1
                self.dense_num += 1
            elif datatype['type'] == 'VecDenseEncoder':
                self.dense_dim += datatype['size']
                self.dense_num += 1
        if self.use_se_net:
            self.se_net = SENet(self.sparse_num)

    def forward(self, x):
        emb_output = []
        dense_output = []
        emb_index = 0
        se_net_input = []
        # se_net_input_dense = []
        for index in range(len(self.datatypes)):
            # for index in range(len(self.embs)):
            datatype = self.datatypes[index]
            # print(datatype)
            #             import pdb
            #             pdb.set_trace()
            if datatype['type'] == 'MultiSparseEncoder':
                vec = self.embs[emb_index](x[:, datatype['index']: datatype['index'] + datatype['size']].int())
                if self.use_se_net:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
                emb_output.append(vec)
                emb_index += 1
            elif datatype['type'] == 'SparseEncoder' or datatype['type'] == 'BucketSparseEncoder':
                vec = self.embs[emb_index](x[:, datatype['index']].int())
                emb_output.append(vec)
                if self.use_se_net:
                    se_net_input.append(torch.mean(vec, dim=1).view(-1, 1))
                emb_index += 1
            elif datatype['type'] == 'DenseEncoder' or datatype['type'] == 'VecDenseEncoder':
                vec = x[:, datatype['index']: datatype['index'] + datatype['size']]
                # if self.use_se_net:
                #     se_net_input_dense.append(torch.mean(vec, dim=1).view(-1, 1).float())
                dense_output.append(vec)

        if len(se_net_input) != 0 and self.use_se_net:
            # import pdb
            # pdb.set_trace()
            se_net_output = self.se_net(torch.cat(se_net_input, dim=1))
            for i in range(self.sparse_num):
                emb_output[i] = emb_output[i] * se_net_output[-1, i:i+1]
            # for i in range(self.dense_num):
            #     dense_output[i] = dense_output[i] * se_net_output[-1, i+self.sparse_num: i+self.sparse_num+1]

        if len(dense_output) != 0 and len(emb_output) != 0:
            output = torch.cat([torch.cat(emb_output, dim=1), torch.cat(dense_output, dim=1)], dim=1).float()
        elif len(dense_output) != 0 and len(emb_output) == 0:
            output = torch.cat(dense_output, dim=1)
        elif len(dense_output) == 0 and len(emb_output) != 0:
            output = torch.cat(emb_output, dim=1)
        else:
            output = torch.empty()

        return output.float()