# coding=utf-8
"""
训练预测DSSM模型的脚本
"""
import argparse
import os
import pickle

import pandas as pd
import torch
from sklearn.utils import shuffle

from script.DSSMTrainable import DSSMTrainable

user_cols = ['UserID','Gender','Age','JobID']
item_cols = ['MovieID', 'ratings_count', 'ratings_mean']


def parse_opt():
    parser = argparse.ArgumentParser()
    # python main.py --data_path dataset/ml-1m/feature --model_path checkpont --whether_train train --epochs 10 --batch_size 2048
    # --dropout 0.0 --lr 0.001 --optimizer Adam --device gpu --encode_min 0.0 --neg_sampling 5 --use_senet True
    # Data Info
    parser.add_argument('--data_path', type=str, default='dataset/ml-1m/feature')
    parser.add_argument('--model_path', type=str, default='checkpoint/')
    parser.add_argument('--whether_train', type=str, default='train')

    # hy-params info
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--encode_min', type=float, default=0.0)
    parser.add_argument('--neg_sampling', type=int, default=5)
    parser.add_argument('--use_senet', type=bool, default=True)

    args = parser.parse_args()
    return args

def negative_sampling(data, sampling_cnt):
    print('Enter to new negative_sampling.')
    pos_data = data[data['label'] == 1][user_cols + item_cols]
    data_user = pos_data[user_cols]
    data_item = data[item_cols]

    print('Enter to get negative item data!')
    # replace为True表示为有放回的随机抽样
    neg_data_item = data_item.sample(sampling_cnt * len(pos_data), replace=True, random_state=None, axis=0)

    print('Enter to get negative user data!')
    neg_data_user = pd.concat([data_user for _ in range(sampling_cnt)], axis=0)
    print('Enter to shuffle data')
    neg_data_item = shuffle(neg_data_item)
    neg_data_user = shuffle(neg_data_user)

    print('Enter to concat user and item data')
    neg_data = pd.concat([neg_data_user.reset_index(drop=True), neg_data_item.reset_index(drop=True)], axis=1)
    neg_data['label'] = 0
    pos_data['label'] = 1

    # 也加入一些曝光未点击的样本
    neg_data_2 = data[data['label'] == 0][user_cols + item_cols].sample(len(pos_data), replace=True, axis=0)

    neg_data_2['label'] = 0
    print(len(pos_data), len(neg_data), len(neg_data_2))

    all_data = pd.concat([pos_data, neg_data, neg_data_2], axis=0)

    # all_data = pd.concat([pos_data, neg_data], axis=0)
    print(all_data['label'].value_counts())
    # 过滤掉相同的数据
    all_data.drop_duplicates(subset=['UserID', 'MovieID'], keep='first', inplace=True)
    return all_data


def load_data():
    x_val, y_val = None, None
    if opt.whether_train == 'train':
        x = pickle.load(open(os.path.join(opt.data_path, 'x_train.p'), 'rb'))
        y = pickle.load(open(os.path.join(opt.data_path, 'y_train.p'), 'rb'))
        x_val = pickle.load(open(os.path.join(opt.data_path, 'x_val.p'), 'rb'))
        y_val = pickle.load(open(os.path.join(opt.data_path, 'y_val.p'), 'rb'))
    else:
        x = pickle.load(open(os.path.join(opt.data_path, 'x_test.p'), 'rb'))
        y = pickle.load(open(os.path.join(opt.data_path, 'y_test.p'), 'rb'))

    if opt.neg_sampling > 0 and opt.whether_train == 'train':
        train_data = pd.concat([x, pd.DataFrame(y, columns=['label'])], axis=1)
        train_data = negative_sampling(train_data, opt.neg_sampling)
        y = train_data['label'].values.reshape(-1, 1).tolist()
        x = train_data[user_cols + item_cols]

    x = format_data(x)
    if x_val is not None:
        x_val = format_data(x_val)

    return x, y, x_val, y_val



def format_data(data):
    for key in user_cols + item_cols:
        data[key] = data[key].astype(str)

    return data


def train(x, y, x_val=None, y_val=None):
    emb_size= 8
    col_params = {}
    for key in user_cols:
        col_params[key] = {
            'belongs': 'user',
            'emb_size': emb_size
        }
    for key in item_cols:
        col_params[key] = {
            'belongs': 'item',
            'emb_size': emb_size
        }
    # 如果有vector的话，example
    # col_params['fine_vec'] = {
    #     'belongs': 'item',
    #     'use_vec': True,
    #     'vec_len': 128,
    # }
    trainable = DSSMTrainable(x, y, col_params, batch_size=opt.batch_size, shuffle=opt.shuffle,
                              label_encoder_rate_min=opt.encode_min, user_dnn_size=(128, 64, 32),
                              item_dnn_size=(128, 64, 32),
                              dropout=opt.dropout, activation='LeakyReLU', use_senet=opt.use_senet,
                              val_x=x_val, val_y=y_val)
    trainable.train(epochs=opt.epochs, val_step=1, device=device, optimizer='Adam', lr=opt.lr,
                          model_path=opt.model_path)
    trainable.save_model(model_path=opt.model_path)


def test(x, y, model_path):
    trainable = DSSMTrainable(x, y, batch_size=opt.batch_size, model_path=model_path)
    auc = trainable.test(device)
    print('Test Auc is {}'.format(auc))


def main():
    x, y, x_val, y_val = load_data()
    if opt.whether_train == 'train':
        train(x, y, x_val=x_val, y_val=y_val)
    else:
        test(x, y, opt.model_path)


if __name__ == '__main__':
    opt = parse_opt()
    device = torch.device('cpu')
    if opt.device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
    main()