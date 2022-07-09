"""
工具类
@Time     :2022/4/11 3:19 PM
@Author   : FuFan
"""
import datetime
import json
import os
import pickle
import sys
from collections import defaultdict

import pandas as pd
import torch
from sklearn import preprocessing
from torch import nn
from tqdm import tqdm
import numpy as np

sys.path.append('../')

DEFAULT_EMB_DIM = 4


def get_datatypes(df, cols_param=None, label_encoder_rate_min=0.01, excepts=None):
    """
    获取数据对应的字典及格式化方式，目前支持String to Sparse和Float to Dense
    :param df: 传入的数据
    :param cols_param: 该cols属于哪个特征列，对应的特征列是什么等等，目前主要用于双塔来区分该列属于哪个塔
    :param label_encoder_rate_min: 数据中每个字段至少出现的次数占比，低于该值的则赋值为默认值

    :param excepts: 不进入计算
    :return: datatype
    """
    if excepts is None:
        excepts = set()
    datatypes = []
    for c in tqdm(set(df.columns) - set(excepts)):
        datatypes.append(
            {
                "name": c,
                "data_type": 'list' if str(pd.api.types.infer_dtype(df[c])) == 'mixed' else str(
                    pd.api.types.infer_dtype(df[c])),
                "rate": pd.notnull(df[c]).sum() / len(df)
            }
        )

    if len(datatypes) == 0:
        print("No datatypes! Please check your data!")
        return None

    # check， 目前datatype仅支持string， floating类型。
    errors = []
    for r in datatypes:
        if r['data_type'] not in ['string', 'floating', 'list']:
            errors.append((r['name'], r['data_type']))

    belongs_cols = {}
    id_cols = []
    vec_cols = {}
    emb_cols = {}
    bucket_cols = {}
    for name in cols_param.keys():
        if 'id' in cols_param[name].keys():
            id_cols.append(name)
        if 'use_vec' in cols_param[name].keys() and cols_param[name]['use_vec']:
            if 'vec_len' not in cols_param[name].keys():
                errors.append((name, 'lack vec_len key'))
                break
            vec_cols[name] = cols_param[name]['vec_len']
        if 'bucket' in cols_param[name].keys() and cols_param[name]['bucket'] is not None:
            bucket_cols[name] = cols_param[name]['bucket']
        if 'belongs' in cols_param[name].keys():
            if cols_param[name]['belongs'] in belongs_cols.keys():
                belongs_cols[cols_param[name]['belongs']].append(name)
            else:
                belongs_cols[cols_param[name]['belongs']] = [name]
        emb_cols[name] = cols_param[name]['emb_size'] if 'emb_size' in cols_param[name].keys() else DEFAULT_EMB_DIM

    if len(errors) != 0:
        print('Some data is not in special type. error data is {}'.format(errors))
        assert len(errors) == 0

    for i in tqdm(range(len(datatypes))):
        r = datatypes[i]
        if r['data_type'] == 'string':
            if id_cols and r['name'] in id_cols:
                # id类特征的处理
                max_len = -1
                mapping = {}
                total = 0
                d = df[r['name']]
                for j in d[pd.notnull(d)].astype(str):
                    if i not in mapping:
                        mapping[j] = 0
                    mapping[j] += 1
                    total += 1

                #             sc = sorted(sc.items(), key = lambda kv:(kv[1], kv[0]))
                mapping = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
                mapping = mapping[:5000]
                mapping = {j[0] for j in mapping if j[0] != '<nan>'}
                mapping = {j: idd + 1 for idd, j in enumerate(mapping)}
                mapping["<nan>"] = 0  ## 0代表None以及不存在的元素
                datatypes[i]['length'] = len(mapping) + 1
                datatypes[i]['type'] = 'SparseEncoder'
                datatypes[i]['mapping'] = pickle.dumps(mapping, protocol=3)
                datatypes[i]['max_len'] = max_len
                datatypes[i]['size'] = 1
                datatypes[i]['emb_dim'] = emb_cols[r['name']] if emb_cols is not None and r[
                    'name'] in emb_cols.keys() else DEFAULT_EMB_DIM
            else:
                max_len = -1
                mapping = {}
                total = 0
                d = df[r['name']]
                for j in d[pd.notnull(d)].astype(str):
                    if j not in mapping:
                        mapping[j] = 0
                    mapping[j] += 1
                    total += 1

                mapping = {j for j in mapping if mapping[j] >= total * label_encoder_rate_min and j != '<nan>'}
                mapping = {j: index + 1 for index, j in enumerate(mapping)}
                mapping['<nan>'] = 0
                datatypes[i]['length'] = len(mapping) + 1
                datatypes[i]['type'] = 'SparseEncoder'
                datatypes[i]['mapping'] = pickle.dumps(mapping, protocol=3)
                datatypes[i]['max_len'] = max_len
                datatypes[i]['size'] = 1
                datatypes[i]['emb_dim'] = emb_cols[r['name']] if emb_cols is not None and r[
                    'name'] in emb_cols else DEFAULT_EMB_DIM
        if r['data_type'] == 'list':
            if vec_cols and r['name'] in vec_cols.keys():

                # 表示当前为Vec序列，不做操作，直接当成Dense
                max_len = -1
                datatypes[i]['type'] = 'VecDenseEncoder'
                datatypes[i]['nan_value'] = 0.0
                datatypes[i]['mapping'] = None
                datatypes[i]['max_len'] = max_len
                datatypes[i]['size'] = vec_cols[r['name']]
            else:
                # 表示当前列为Multi_hot
                max_len = -1
                mapping = {}
                total = 0
                d = df[r['name']]
                for j in d[pd.notnull(d)]:
                    items = j
                    if max_len < len(items):
                        max_len = len(items)
                    # max_len = len(items)
                    for item in items:
                        if item not in mapping:
                            mapping[item] = 0
                        mapping[item] += 1
                        total += 1
                    # if i not in sc: sc[i] = 0
                    # sc[i] +=
                    # total += 1
                mapping = {j for j in mapping if mapping[j] >= total * label_encoder_rate_min and j != '<nan>'}
                mapping = {j: idd + 1 for idd, j in enumerate(mapping)}
                mapping["<nan>"] = 0  ## 0代表None以及不存在的元素
                datatypes[i]['length'] = len(mapping) + 1
                datatypes[i]['type'] = 'MultiSparseEncoder'
                datatypes[i]['mapping'] = pickle.dumps(mapping, protocol=3)
                datatypes[i]['max_len'] = max_len
                datatypes[i]['size'] = max_len
                datatypes[i]['emb_dim'] = emb_cols[r['name']] if emb_cols is not None and r[
                    'name'] in emb_cols else DEFAULT_EMB_DIM
        elif r['data_type'] == 'floating':
            if r['name'] in bucket_cols.keys():
                # 将floating的数据进行分桶
                max_len = -1
                bucket_freq = [float(item) for item in bucket_cols[r['name']].split(',')]
                bucket = df[r['name']].quantile(bucket_freq).tolist()
                mapping = {}
                for j in range(len(bucket)) :
                    if j == 0:
                        mapping['0.0, ' + str(bucket[j])] = j
                    elif j+1 == len(bucket):
                        mapping[str(bucket[j - 1]) + ',' + str(bucket[j] + 1)] = j
                    else:
                        mapping[str(bucket[j - 1]) + ',' + str(bucket[j])] = j
                # mapping = {j + 1 : [bucket[j], bucket[j+1]] for j in range(len(bucket) - 1)}
                # mapping["<nan>"] = 0  ## 0代表None以及不存在的元素
                datatypes[i]['length'] = len(mapping) + 1
                datatypes[i]['type'] = 'BucketSparseEncoder'
                datatypes[i]['mapping'] = pickle.dumps(mapping, protocol=3)
                datatypes[i]['max_len'] = max_len
                datatypes[i]['size'] = 1
                datatypes[i]['emb_dim'] = emb_cols[r['name']] if emb_cols is not None and r[
                    'name'] in emb_cols else DEFAULT_EMB_DIM
            else:
                max_len = -1
                mapping = preprocessing.MinMaxScaler()
                mapping.fit(df[r['name']].to_numpy().reshape(-1, 1))
                datatypes[i]['type'] = 'DenseEncoder'
                datatypes[i]['nan_value'] = 0.0
                datatypes[i]['mapping'] = pickle.dumps(mapping, protocol=3)
                datatypes[i]['max_len'] = max_len
                datatypes[i]['size'] = 1

        flag = True
        if belongs_cols is not None:
            for key in belongs_cols.keys():
                if r['name'] in belongs_cols[key]:
                    datatypes[i]['belongs'] = key
                    flag = False
                    break
        if flag:
            datatypes[i]['belongs'] = '<nan>'

    datatypes = sorted(datatypes, key=lambda x: (x['type'], x['name']), reverse=True)
    # 设置index和size
    index_belongs = {}
    for index in range(len(datatypes)):
        if datatypes[index]['belongs'] not in index_belongs.keys():
            index_belongs[datatypes[index]['belongs']] = 0
        datatypes[index]['index'] = index_belongs[datatypes[index]['belongs']]
        index_belongs[datatypes[index]['belongs']] += datatypes[index]['size']
    # datatypes.sort_values('type', inplace=True)
    return datatypes


def data_to_list(df, datatypes):
    """
    将pandas的数据转成二维数组，并格式化
    :param df:
    :param datatypes:
    :return:
    """
    res = [[] for _ in range(len(df))]
    # raw_index = 0
    for datatype in datatypes:
        col_name = datatype['name']
        mapping = pickle.loads(datatype['mapping']) if datatype['mapping'] else None
        print(col_name)
        data = df[col_name].tolist()
        row_index = 0
        for item in data:
            if datatype['type'] == 'SparseEncoder':
                try:
                    res[row_index].append(mapping.get(item, 0))
                    row_index += 1
                except:
                    res[row_index].append(0)
                    row_index += 1
                    # import pdb
                    # pdb.set_trace()

                # input.append(mapping.get(item[col_name], 0))
            elif datatype['type'] == 'MultiSparseEncoder':
                for i in range(datatype['size']):
                    if i < len(item):
                        res[row_index].append(mapping.get(item[i], 0))
                    else:
                        res[row_index].append(0)
                row_index += 1
            elif datatype['type'] == 'DenseEncoder':
                nan_value = datatype['nan_value']
                data_min, data_max = mapping.data_min_[0], mapping.data_max_[0]
                if item is None or np.isnan(item):
                    res[row_index].append(nan_value)
                else:
                    if data_min == data_max:
                        res[row_index].append(nan_value)
                    else:
                        res[row_index].append((item - data_min) / (data_max - data_min))
                row_index += 1
            elif datatype['type'] == 'VecDenseEncoder':
                for i in range(datatype['size']):
                    if i < len(item):
                        res[row_index].append(item[i])
                row_index += 1
            elif datatype['type'] == 'BucketSparseEncoder':
                flag = False
                for key in mapping.keys():
                    start, end = key.split(',')
                    if float(start) <= item < float(end):
                        res[row_index].append(mapping[key])
                        flag = True
                        break
                if not flag:
                    res[row_index].append(0)

                row_index += 1

    return np.asarray(res)


def data_to_type_list(df, datatypes):
    """
    供双塔使用，将每个类别的数据转为二维数组，并格式化
    :param df:
    :param datatypes:
    :return:
    """
    res = {}
    datatype_dict = {}
    for r in datatypes:
        if r['belongs'] not in datatype_dict.keys():
            datatype_dict[r['belongs']] = [r]
        else:
            datatype_dict[r['belongs']].append(r)
    for key in datatype_dict.keys():
        print('Start to change data in {} group'.format(key))
        res[key] = data_to_list(df, datatype_dict[key])

    print('End to data type list!')
    return res


def to_list(x, max_len):
    if x and x is not None and x != '<nan>':
        x_list = str(x).split(',')
        if len(x_list) < max_len:
            for _ in range(max_len - len(x_list)):
                x_list.append('<nan>')
        else:
            x_list = x_list[:max_len]
        return x_list
    else:
        return ['<nan>' for _ in range(max_len)]


def get_max_len(df):
    max_len = 0
    for x in df.tolist():
        if x and x != None and x != '<nan>':
            length = len(x.split(','))
            if max_len < length:
                max_len = length
    return max_len


def get_max_len_from_datatype(datatype, name):
    for x in datatype:
        if x['name'] == name:
            return datatype['max_len']

    return -1


def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == 'leakyrelu':
            return nn.LeakyReLU()
        else:
            return getattr(nn, activation)()
    else:
        return


def get_optimizer(optimizer, params, lr):
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    except:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer


