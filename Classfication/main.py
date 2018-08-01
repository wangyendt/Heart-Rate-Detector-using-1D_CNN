#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/08/01 12:52
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : main.py
# @Software: PyCharm

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torch.autograd import Variable


def load_data():
    path = '../Preprocessing/output_matrix'
    dataset_x = []
    dataset_y = []
    for f in os.listdir(path):
        data = np.loadtxt(os.path.join(path, f))
        dataset_x.extend(data)
        if 'pos' in f:
            dataset_y.extend(np.ones([data.shape[0], 1]))
        if 'neg' in f:
            dataset_y.extend(np.zeros([data.shape[0], 1]))
    return np.array(dataset_x), np.array(dataset_y)


class CNN_1D(torch.nn.Module):
    def __init__(self):
        super(CNN_1D, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, 3)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.maxpool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(64, 128, 3)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.maxpool2 = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear(896, 224)
        self.fc2 = torch.nn.Linear(224, 56)
        self.fc3 = torch.nn.Linear(56, 2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.maxpool2(out)
        out = out.view(-1, self.num_flat_features(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    X, y = load_data()
    z = list(zip(X, y))
    np.random.shuffle(z)
    X_, y_ = zip(*z)
    X_ = np.array(X_)
    y_ = np.array(y_)
    gap = int(0.7 * len(z))
    train_X = Variable(torch.FloatTensor(X_[:gap][:, np.newaxis, :])).cuda()
    train_y = Variable(torch.LongTensor(y_[:gap].squeeze())).cuda()
    test_X = Variable(torch.FloatTensor(X_[gap:][:, np.newaxis, :]), requires_grad=False).cuda()
    test_y = Variable(torch.LongTensor(y_[gap:].squeeze()), requires_grad=False).cuda()
    cnn_1d = CNN_1D().cuda()
    epochs = 100
    lr = 0.01
    optimizer = torch.optim.Adam(cnn_1d.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    test_pred = cnn_1d(test_X)
    print(test_pred)
    print(np.argmax(test_pred.cpu().data, axis=1))
    print(test_y)
    err = test_y.cpu().numpy() == np.argmax(test_pred.cpu().data, axis=1)
    err = np.sum(err.numpy()) / np.shape(test_y.data)[0]
    print('Accuracy: ', 100 * err, '%')

    for epoch in range(epochs):
        pred = cnn_1d(train_X)
        optimizer.zero_grad()
        loss = criterion(pred, train_y)
        if epoch % 10 == 0:
            print(epoch, loss.data / train_X.size()[0])
        losses.append(loss.data / train_X.size()[0])
        loss.backward()
        optimizer.step()

    test_pred = cnn_1d(test_X)
    # print(test_pred)
    # print(np.argmax(test_pred.cpu().data, axis=1))
    # print(test_y)
    err = (test_y.cpu().numpy() == np.argmax(test_pred.cpu().data, axis=1)).numpy()
    # print(err)
    err_ind = np.where(err == 0)[0]
    for ii in range(len(err_ind)):
        test_x_numpy = test_X.cpu().numpy().squeeze()
        test_y_numpy = test_y.cpu().numpy().squeeze()
        print(np.shape(test_x_numpy))
        plt.plot(test_x_numpy[ii])
        plt.title(test_y_numpy[ii])
        plt.show()
    # print(np.where(err == 0)[0])
    err = np.sum(err) / np.shape(test_y.data)[0]
    print('Accuracy: ', 100 * err, '%')
