#-*- coding : utf-8-*-
# coding:unicode_escape

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import re
import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

"""
数据预处理
对原始电影数据，user数据进行处理
"""
def user_data_processing(path):
    '''
    对原始user数据进行处理
    UserID：保持不变
    JobID：保持不变
    Gender字段：需要将‘F’和‘M’转换成0和1。
    Age字段：要转成7个连续数字0~6。
    舍弃： zip-code
    '''
    
    print('user_data_processing....')
    user_title = ['UserID','Gender','Age','JobID','Zip-code']
    print(path)
    users = pd.read_table(os.path.join(path, 'users.dat'), sep='::', header=None,
        names=user_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values #a list

    gender_to_int = {'F':0,'M':1}
    users['Gender'] = users['Gender'].map(gender_to_int)
    age2int = {val:ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age2int)

    return users, users_orig

def movie_data_processing(path, title_length = 16):
    '''
    对原始movie数据不作处理
    Genres字段：进行int映射，因为有些电影是多个Genres的组合,需要再将每个电影的Genres字段转成数字列表.
    Title字段：首先去除掉title中的year。然后将title映射成数字列表。（int映射粒度为单词而不是整个title）
    Genres和Title字段需要将长度统一，这样在神经网络中方便处理。
    空白部分用‘< PAD >’对应的数字填充。
    '''
    print('movie_data_processing....')
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table(os.path.join(path, 'movies.dat'), sep='::', encoding='ISO-8859-1',
        header=None, names=movies_title, engine='python')
    movies_orig = movies.values#length:3883
    # title处理，首先将year过滤掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_re_year = {val:pattern.match(val).group(1) for val in set(movies['Title'])}
    movies['Title'] = movies['Title'].map(title_re_year)
    #title的int映射
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    title_set.add('PADDING')
    title2int = {val: ii for ii, val in enumerate(title_set)}  # length:5215

    # 构建title_map，每个title映射成一个int list，然后对于长度不足16的使用pad进行补全
    title_map = {val: [title2int[row] for row in val.split()] \
                 for val in set(movies['Title'])}
    for key in title_map.keys():
        padding_length = title_length - len(title_map[key])
        padding = [title2int['PADDING']] * padding_length
        title_map[key].extend(padding)
        # for cnt in range(title_length - len(title_map[key])):
        #     title_map[key].insert(len(title_map[key]) + cnt, title2int['PADDING'])
    movies['Title'] = movies['Title'].map(title_map)
    print(len(movies['Title'][0]))

    #电影类型转为数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('PADDING')
    genres2int = {val:ii for ii, val in enumerate(genres_set)} # length:19

    #和title的处理相同，对每个电影的genres构建一个等长的int list映射
    genres_map={val:[genres2int[row] for row in val.split('|')]\
            for val in set(movies['Genres'])}
    # for key in genres_map:
    #     padding_length = len(genres_set) - len(genres_map[key])
    #     padding = [genres2int['PADDING']] * padding_length
    #     genres_map[key].extend(padding)
        # for cnt in range(max(genres2int.values()) - len(genres_map[key])):
        #     genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(genres_map)

    return movies, movies_orig, genres2int, title_set


def rating_data_processing(path):
    '''
    rating数据处理，只需要将timestamps舍去，保留其他属性即可
    '''
    print('rating_data_processing....')
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table(os.path.join(path, 'ratings.dat'), sep='::',
        header=None, names=ratings_title, engine='python')

    # 将同一电影的ratings进行求和平均并赋值给各个电影
    ratings_mean = ratings.groupby('MovieID')['ratings'].mean().astype('int')
    ratings_counts = ratings.groupby('MovieID')['ratings'].size()
    # print(ratings_counts)
    # print('-------------------------------------')
    # 将评论数据进行分桶, 分为5个等级
    ratings_counts_max = max(ratings_counts)
    # print(ratings_counts_max)
    cut_num = int(ratings_counts_max / 5) + 1
    cut_range = []
    for i in range(5 + 1):
        cut_range.append(i * cut_num)
    # print(cut_range)
    ratings_counts = pd.cut(ratings_counts, bins=cut_range, labels=False)
    # print(ratings_counts)

    if len(ratings_mean) != len(ratings_counts):
        print('total_ratings is not equal ratings_counts!')
    else:
        ratings = pd.merge(pd.merge(ratings, ratings_counts, on='MovieID'), ratings_mean, on='MovieID')
        # rename the columns
        # ratings_x: 原ratings
        # ratings_y: ratings_counts
        # ratings: ratings_mean
        ratings = ratings.rename(columns={'ratings': 'ratings_mean'}).rename(columns={'ratings_x': 'ratings'}).rename(columns={'ratings_y': 'ratings_count'})
        ratings = ratings.filter(regex='UserID|MovieID|ratings_mean|ratings_count|ratings')

    rating_datatype = []
    rating_datatype.append({'name': 'ratings_count', 'len': ratings['ratings_count'].max() + 1,
                           'ori_scaler': cut_range,
                           'type': 'LabelEncoder', 'nan_value': None})
    rating_datatype.append({'name': 'ratings_mean', 'len': ratings['ratings_mean'].max() + 1,
                            'ori_scaler': {i: i for i in range(ratings['ratings_mean'].max() + 1)},
                            'type': 'LabelEncoder', 'nan_value': None})
    return ratings, rating_datatype

def get_feature():
    """
    将多个方法整合在一起，得到movie数据，user数据，rating数据。
    然后将三个table合并到一起，组成一个大table。
    最后将table切割，分别得到features 和 target（rating）
    """
    title_length = 16
    path = os.path.abspath(os.path.join('../../', 'dataset/ml-1m'))
    users, users_orig = user_data_processing(path)
    movies, movies_orig, genres2int,title_set = movie_data_processing(path)
    ratings, rating_datatype = rating_data_processing(path)

    #merge three tables
    data = pd.merge(pd.merge(ratings, users), movies)

    # split data to feature set:X and lable set:y
    target_fields = ['ratings']
    features, tragets_pd = data.drop(target_fields, axis=1), data[target_fields]
    # features = feature_pd.values


    # 针对ratings进行数据的分割，将ratings大于等于3的作为用户click的数据，反之为不会click的数据
    tragets_pd.ratings[tragets_pd['ratings'] <= 3] = 0
    tragets_pd.ratings[tragets_pd['ratings'] > 3] = 1

    targets = tragets_pd.values

    # 将处理后的数据保存到本地
    if not os.path.exists(os.path.join(path, 'feature')):
        os.makedirs(os.path.join(path, 'feature'))
    f = open(os.path.join(path, 'feature/ctr_features.p'), 'wb')
    # ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres' 'ratings_counts' 'ratings_mean']
    pickle.dump(features, f)

    f = open(os.path.join(path, 'feature/ctr_target.p'), 'wb')
    pickle.dump(targets, f)

    f = open(os.path.join(path, 'feature/ctr_params.p'), 'wb')
    pickle.dump((title_length, title_set, genres2int, features, targets, \
                 ratings, users, movies, data, movies_orig, users_orig), f)

    f = open(os.path.join(path, 'feature/ctr_data.p'), 'wb')
    pickle.dump(data, f)

    return features, targets, data


def split_train_test(feature, targets):
    """
    将feature和targets分割成train, val, test。
    并将数据处理成两类，一种为onehot形式，一种为数据流形式
    :param feature:
    :param targets:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(feature, targets, test_size=0.2, random_state=2022)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2022)

    x_train.reset_index(drop=True, inplace=True)
    # y_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    # y_test.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)
    # y_val.reset_index(drop=True, inplace=True)

    # 保存为数据流格式, 对于seq和multi的数据，暂不处理
    path = os.path.join('../../', 'dataset/ml-1m')
    f = open(os.path.join(path, 'feature/x_train.p'), 'wb')
    pickle.dump(x_train, f)

    f = open(os.path.join(path, 'feature/y_train.p'), 'wb')
    pickle.dump(y_train, f)

    f = open(os.path.join(path, 'feature/x_test.p'), 'wb')
    pickle.dump(x_test, f)

    f = open(os.path.join(path, 'feature/y_test.p'), 'wb')
    pickle.dump(y_test, f)

    f = open(os.path.join(path, 'feature/x_val.p'), 'wb')
    pickle.dump(x_val, f)

    f = open(os.path.join(path, 'feature/y_val.p'), 'wb')
    pickle.dump(y_val, f)


def onehot_format(feature, datatype, cols):
    res = []
    for _, r in tqdm(datatype[datatype['type'] == "LabelEncoder"].iterrows()):
        if r['name'] not in cols:
            continue
        # sc = pickle.loads(r.scaler)
        d = feature[r['name']]
        assert r['len'] > d.max()
        onehot = np.zeros((len(d), r['len']), dtype=np.int)
        onehot[np.arange(len(d)), d.astype(int)] = 1
        res.append(onehot)
    for _, r in tqdm(datatype[datatype['type'] == "MinMaxScaler"].iterrows()):
        if r['name'] not in cols:
            continue
        sc = pickle.loads(r['scaler'])
        # sc = MinMaxScaler()
        v = feature[r['name']].reshape((-1, 1))
        mask = np.isnan(v)
        v[~mask] = sc.transform(v[~mask].reshape((-1, 1))).reshape(-1)
        v[mask] = r['nan_value']
        res.append(v)
    for _, r in tqdm(datatype[datatype['type'] == "MultiLabelEncoder"].iterrows()):
        if r['name'] not in cols:
            continue
        d = feature[r['name']]
        onehot = np.zeros((len(d), r['len']), dtype=np.int)
        for index in range(len(d)):
            for item in d[index]:
                onehot[index, item] = 1
            # for item in d[index].split(','):
            #     onehot[index, item] = 1
        res.append(onehot)
    res = np.concatenate(res, axis=1)
    return res


def datanorm_xlearn(df, datatypes, tqdm=lambda x, *args, **kargs: x):
    """
    老版本 xlearn输入格式化
    :param df: 喂入过DeepFM后出来的数据
    :param datatypes:
    :param tqdm:
    :return:
    """
    features = []
    for _, r in tqdm(datatypes[datatypes.type == "LabelEncoder"].iterrows()):
        sc = pickle.loads(r.scaler)
        d = df[r.name]
        assert len(sc) > d.max()
        onehot = np.zeros((len(d), len(sc)))
        onehot[np.arange(len(d)), d.astype(int)] = 1
        features.append(onehot)
    for _, r in tqdm(datatypes[datatypes.type == "MinMaxScaler"].iterrows()):
        sc = pickle.loads(r.scaler)
        v = df[r.name].reshape((-1, 1))
        mask = np.isnan(v)
        v[~mask] = sc.transform(v[~mask].reshape((-1, 1))).reshape(-1)
        v[mask] = r.nan_value
        features.append(v)
    for _, r in tqdm(datatypes[datatypes.type == "keep"].iterrows()):
        v = df[r.name].reshape((-1, 1))
        features.append(v)
    features = np.concatenate(features, axis=1)
    return features

def main():
    features, targets, data = get_feature()
    split_train_test(features, targets)

if __name__ == '__main__':
    main()