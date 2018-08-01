#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/07/30 17:07
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : LoadData.py
# @Software: PyCharm

import os
import numpy as np


def load_pos_data(root):
    files = os.listdir(root)
    for f in files:
        if '.txt' in f and 'pos' in f:
            data = np.loadtxt(os.path.join(root, f))
            yield (data, f)


def load_neg_data(root):
    files = os.listdir(root)
    for f in files:
        if '.txt' in f and 'neg' in f:
            data = np.loadtxt(os.path.join(root, f))
            yield (data, f)


if __name__ == '__main__':
    data = np.loadtxt('rawdata/data1.txt')[:, 0]
    data2 = -np.loadtxt('rawdata/data2.txt')[131:3480, 0]
    import Preprocessing.RemoveBaseline as rb

    br = rb.BaselineRemoval()

    pos = {
        1: data[13146:16307],
        2: data[16334:20630],
        3: data[20662:22503],
        4: data2
    }
    neg = {
        1: data[:6585],
        2: data[10630:13145],
        3: data[22503:]
    }
    import matplotlib.pyplot as plt

    # plt.plot(pos[1])
    # plt.plot(br.baseline_removing(pos[1]))
    # plt.plot(pos[1]-br.baseline_removing(pos[1]))
    # plt.show()

    root = 'classified data'

    [np.savetxt(''.join((root, '/pos', str(i), '.txt')), pos[i] - br.baseline_removing(pos[i])) for i in range(1, 5)]
    [np.savetxt(''.join((root, '/neg', str(i), '.txt')), neg[i]) for i in range(1, 4)]

    [(plt.plot(np.loadtxt(''.join((root, '/pos', str(i), '.txt')))), plt.show()) for i in range(1, 5)]
