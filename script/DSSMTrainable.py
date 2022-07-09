"""
DSSM训练class
@TIME       :17/05/2022 16:25
@Author     : Heart Fu
"""
import pickle
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.DSSMDataSet import DSSMDataSet
from model.DSSM import DSSM
from utils.utils import get_optimizer


class DSSMTrainable:
    def __init__(self, x, y, cols_params=None, batch_size=2048, shuffle=True, val_x=None, val_y=None,
                 label_encoder_rate_min=0.0001, user_dnn_size=(64, 32), item_dnn_size=(64, 32), dropout=0.2,
                 activation='LeakyReLU', model_path=None, use_senet=False, device='cpu'):

        self.use_senet = use_senet
        self.best_weight = None

        if model_path is not None:
            # test or eval class
            self.model = torch.load(os.path.join(model_path, 'best_train_model.pt'),
                                    map_location=torch.device(device))
            # file = open(os.path.join(model_path, 'datatype.pkl'), 'rb')
            # self.datatype = pickle.load(file)
            self.datatypes = self.model.user_datatypes + self.model.item_datatypes
            self.test_dataset = DSSMDataSet(x, np.array(y).reshape(-1, 1), datatype=self.datatypes,
                                            label_encoder_rate_min=label_encoder_rate_min)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
            self.trainable = False
        else:
            self.trainable = True
            # file = open(os.path.join('/home/fufan/', 'datatype.pkl'), 'rb')
            # self.datatype = pickle.load(file)
            # train class
            self.train_dataset = DSSMDataSet(x, np.array(y).reshape(-1, 1), cols_params,
                                             label_encoder_rate_min=label_encoder_rate_min,
                                             neg_sampling=neg_sampling, training=True)
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
            print('Finish train dataloader, len is {}'.format(len(self.train_dataset)))
            self.datatypes = self.train_dataset.datatypes
            if val_x is not None and val_y is not None:
                self.val_dataset = DSSMDataSet(val_x, np.array(val_y).reshape(-1, 1), datatype=self.datatypes)
                self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size)
            else:
                self.val_dataloader = None
                self.datatypes = self.train_dataset.datatypes

            user_datatypes = []
            item_datatypes = []
            for datatype in self.datatypes:
                if datatype['belongs'] == 'user':
                    user_datatypes.append(datatype)
                else:
                    item_datatypes.append(datatype)
            # if model_path:
            # self.model = torch.load(os.path.join('/home/fufan/', 'model.pt'))

            # else:
            self.model = DSSM(user_datatypes, item_datatypes, user_dnn_size=user_dnn_size, item_dnn_size=item_dnn_size,
                              dropout=dropout, activation=activation, use_senet=self.use_senet)
            # # torch.save(self.model, os.path.join('/home/fufan/', 'model.pt'))
            # self.save_model('/home/fufan/')

        print(self.model)
        # model = self.model.to(device)

    def set_lr(self, optimizer, decay_factor):
        for group in optimizer.param_groups:
            group['lr'] = group['lr'] * decay_factor

    def train(self, epochs=10, val_step=1, device='cpu', optimizer='Adam', lr=1e-5, model_path=None,
              lr_decay_rate=0, lr_decay_step=0):
        # writer = SummaryWriter()
        optimizer = get_optimizer(optimizer, self.model.parameters(), lr)
        loss_func = nn.BCELoss()
        loss_func = loss_func.to(device)
        self.model = self.model.to(device)
        best_val_auc, best_val_epoch = 0, 0
        train_auc = 0
        best_train_auc = 0
        for epoch in range(epochs):
            self.model.train()

            if lr_decay_rate != 0 and lr_decay_step !=0 and epoch % lr_decay_rate == 0:
                self.set_lr(optimizer, lr_decay_rate)

            total_loss = 0
            total_len = 0

            train_pre = list()
            train_label = list()

            progress_bar = tqdm(self.train_dataloader, desc='|Train Epoch {}'.format(epoch), leave=False)
            for index, data in enumerate(progress_bar):
            # for index, data in enumerate(self.train_dataloader):
                user_x, item_x, y = data
                user_x, item_x, y = user_x.to(device).float(), item_x.to(device).float(), y.to(device).float()
                user_emb, item_emb = self.model(user_x, item_x)

                y_pre = torch.sigmoid((user_emb * item_emb).sum(dim=-1)).reshape(-1, 1)

                train_pre.extend(y_pre.cpu().detach().numpy())
                train_label.extend(y.cpu().detach().numpy())

                loss = loss_func(y_pre, y)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                total_loss += loss.item() * len(y)
                total_len += len(y_pre)
                progress_bar.set_postfix({'loss': loss.item()}, refresh=True)

            train_loss, train_auc = self.eval(self.train_dataloader, device, loss_func)
            if train_auc > best_train_auc:
                torch.save(self.model, os.path.join(model_path, 'best_train_model.pt'))
                best_train_auc = train_auc
                self.best_weight = self.model.state_dict()
            if self.val_dataloader is not None and epoch % val_step == 0:
                val_loss, val_auc = self.eval(self.val_dataloader, device, loss_func=loss_func)
                # val_loss, val_auc = eval_DSSM(model, val_dataloader, loss_func, device)
                if val_auc > best_val_auc and model_path is not None:
                    best_val_auc = val_auc
                    torch.save(self.model, os.path.join(model_path, 'best_model.pt'))
                print("epoch:{}, train loss:{:.5}, train auc:{:.5}, val loss:{:.5}, val auc:{:.5}".format(epoch,
                                                                                                          train_loss,
                                                                                                          train_auc,
                                                                                                          val_loss,
                                                                                                          val_auc))
            else:
                print("epoch:{}, train loss:{:.5}, train auc:{:.5}".format(epoch, train_loss, train_auc))

        return train_auc

    def eval(self, dataloader, device='cpu', loss_func=None):
        self.model.eval()
        labels, pre_pro = list(), list()
        val_loss = 0
        val_len = 0
        index = 0
        with torch.no_grad():
            for data in dataloader:
                index += 1
                user_x, item_x, y = data
                user_x, item_x, y = user_x.to(device).float(), item_x.to(device).float(), y.to(device).float()
                user_emb, item_emb = self.model(user_x, item_x)
                y_pre = torch.sigmoid((user_emb * item_emb).sum(dim=-1)).reshape(-1, 1)

                pre_pro.extend(y_pre.cpu().detach().numpy())
                labels.extend(y.cpu().detach().numpy())

                if loss_func:
                    loss_val = loss_func(y_pre, y)
                    val_loss += loss_val.item() * len(y)
                val_len += len(y)

        auc = roc_auc_score(np.array(labels), np.array(pre_pro))
        loss = val_loss / val_len
        return loss, auc

    def test(self, device='cpu'):
        self.model = self.model.to(device)
        _, test_auc = self.eval(self.test_dataloader, device)
        print('test auc: {}'.format(test_auc))
        return test_auc

    def save_model(self, model_path):
        torch.save(self.model, os.path.join(model_path, 'model.pt'))
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model_param.pkl'))
        with open(os.path.join(model_path, 'datatype.pkl'), 'wb') as f:
            pickle.dump(self.datatypes, f)
