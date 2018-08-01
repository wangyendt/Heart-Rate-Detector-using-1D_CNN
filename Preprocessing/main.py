#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/08/01 13:21
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : main.py
# @Software: PyCharm

from Preprocessing import LoadData
import numpy as np

if __name__ == '__main__':
    pos_datas = LoadData.load_pos_data('preprocessing output')
    neg_datas = LoadData.load_neg_data('preprocessing output')
    pos_data_mat = [data for (data, f) in pos_datas]
    neg_data_mat = [data for (data, f) in neg_datas]
    print(np.shape(pos_data_mat))
    print(np.shape(neg_data_mat))
    np.savetxt('output_matrix/pos_mat.txt', pos_data_mat, '%f')
    np.savetxt('output_matrix/neg_mat.txt', neg_data_mat, '%f')
