#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/08/01 01:57
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : Resize.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from Preprocessing import LoadData


def resize_array(arr, dst_len):
    src_len = len(arr)
    f = interpolate.interp1d(np.linspace(0, dst_len, src_len), arr)
    arr_ = f(range(dst_len))
    # plt.plot(arr)
    # plt.plot(arr_)
    # plt.show()
    return arr_


def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


root = 'data slices'


def resize_pos():
    data_len = [len(np.loadtxt(os.path.join(root, f))) if 'pos' in f else 0 for f in os.listdir(root)]
    num_pos_files = sum(['pos' in f for f in os.listdir(root)])
    average_len = np.round(np.sum(data_len) / num_pos_files).astype(np.int)
    print(average_len)
    print('#pos: ', sum(['pos' in f for f in os.listdir(root)]))
    datas = LoadData.load_pos_data('data slices')
    for (data, f) in datas:
        arr_ = normalize_array(resize_array(data, average_len))
        np.savetxt(os.path.join('preprocessing output',
                                f),
                   arr_,
                   fmt='%f')


def resize_neg():
    data_len = [len(np.loadtxt(os.path.join(root, f))) if 'neg' in f else 0 for f in os.listdir(root)]
    num_pos_files = sum(['neg' in f for f in os.listdir(root)])
    average_len = np.round(np.sum(data_len) / num_pos_files).astype(np.int)
    print(average_len)
    print('#neg: ', sum(['neg' in f for f in os.listdir(root)]))
    datas = LoadData.load_neg_data('data slices')
    for (data, f) in datas:
        arr_ = normalize_array(resize_array(data, average_len))
        np.savetxt(os.path.join('preprocessing output',
                                f),
                   arr_,
                   fmt='%f')


if __name__ == '__main__':
    resize_pos()
    resize_neg()
