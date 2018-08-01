#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/07/30 18:00
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : SliceECG.py
# @Software: PyCharm

from Preprocessing import LoadData
import matplotlib.pyplot as plt
import numpy as np


class DataProcessing:
    def __init__(self, data):
        self.data = data
        self.base = data
        self.force_signal = data
        self.energy = None
        self.flag = None
        self.mov_avg_len = 5
        self.alpha = 3
        self.beta = 5
        self.energy_thd = 8  # 8 for 'pos4', 100 for others
        self.min_td_time = 0
        self.min_tu_time = 0
        self.upper = self.energy_thd
        self.lower = self.energy_thd
        self.step_u = 0
        self.step_l = 0

    def pre_process(self):
        if np.ndim(self.data) == 1:
            # self.data = self.data - self.data[0]
            self.data = self.data[:, np.newaxis]
        else:
            pass
            # self.data = self.data - self.data[0, :]

    def calc_moving_avg(self):
        output = []
        for ii in range(len(self.data)):
            if ii <= self.mov_avg_len:
                output.append(np.mean(self.data[:ii + 1, :], 0))
            else:
                output.append(np.mean(self.data[ii - (self.mov_avg_len - 1):ii + 1, :], 0))
        self.data = np.array(output)

    def calc_energy(self):
        self.force_signal = np.array(self.force_signal)
        if self.force_signal.ndim == 1:
            self.force_signal = self.force_signal[:, np.newaxis]
        m = self.force_signal.shape[0]
        energy_n = np.zeros((m, 1))
        for ii in range(m - self.alpha - self.beta):
            energy_n[ii + self.alpha + self.beta] = \
                1 / self.alpha * \
                np.sum(np.sum(np.abs(
                    self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] -
                    self.force_signal[ii:ii + self.alpha, :]), 1), 0)
            diff_mat = self.force_signal[ii + self.beta:ii + self.beta + self.alpha, :] - self.force_signal[
                                                                                          ii:ii + self.alpha, :]
            diff_mat_max_sub = np.array(np.where(np.abs(diff_mat) == np.max(np.abs(diff_mat))))
            diff_mat_max_sub = diff_mat_max_sub[:, 0]
            max_sign = np.sign(diff_mat[diff_mat_max_sub[0], diff_mat_max_sub[1]])
            energy_n[ii + self.alpha + self.beta] *= max_sign
        self.energy = energy_n.T[0]

    # t is tvalue < thd
    @staticmethod
    def update_flag_status(f, r, t, tdf, tuf):
        f_ = f ^ r and f or not (f ^ r) and not (f ^ t)
        f_ = not f and f_ and tuf or f and (f_ or not tdf)
        r_ = not r and f and t or r and not (not f and t)
        return f_, r_

    def calc_flag(self):
        self.flag = np.zeros(self.energy.shape, dtype=np.bool)
        ready = False
        touch_down_frm = 0
        touch_up_frm = self.min_tu_time + 1
        for ii in range(1, self.flag.shape[0]):
            f = bool(self.flag[ii - 1])
            t = (not f and (self.energy[ii] < self.energy_thd)) or (f and (self.energy[ii] > self.energy_thd * 0.7))
            touch_down_frm = touch_down_frm + 1 if self.flag[ii - 1] else 0
            touch_up_frm = touch_up_frm + 1 if not self.flag[ii - 1] else 0
            tdf = touch_down_frm >= self.min_td_time
            tuf = touch_up_frm >= self.min_tu_time
            self.flag[ii], ready = self.update_flag_status(f, ready, t, tdf, tuf)
        self.flag = np.array(self.flag, dtype=np.int)

    def calc_valleys(self, f):
        tds = np.where(np.diff(self.flag) == 1)[0]
        sub_data = np.vsplit(self.data, tds)[:-1]
        closest_valley_inds = np.array(
            [len(sd) - (np.where(np.diff(np.sign(np.diff(sd, axis=0)), axis=0) > 0)[0] + 1)[-1] for sd in sub_data]
        )
        slices = np.vsplit(self.data, tds - closest_valley_inds)
        for ind, slc in enumerate(slices):
            if ind == 0:
                continue
            np.savetxt(''.join(('data slices/', f[:-4], '_', str(ind), '.txt')), slc, fmt='%f')
        fig = plt.figure()
        fig.set_size_inches(60, 10)
        plt.plot(self.data)
        plt.plot(tds, self.data[tds], 'o')
        plt.plot(tds - closest_valley_inds, self.data[tds - closest_valley_inds], 'o')
        plt.title(f)
        plt.xlabel('Time Series')
        plt.ylabel('ADC')
        plt.legend(('rawdata', 'flag point', 'valleys'))
        plt.show()

    def show_fig(self, f):
        fig = plt.figure()
        fig.set_size_inches(60, 10)
        plt.subplot(211)
        plt.plot(self.data)
        plt.title('rawdata')
        plt.legend(tuple([''.join(('rawdata', str(ii))) for ii in range(np.shape(self.data)[1])]))
        plt.ylabel('ADC')
        plt.subplot(212)
        # plt.plot(self.force_signal, '-', linewidth=3)
        plt.plot(self.energy)
        plt.hlines(self.upper, 0, self.data.shape[0], linestyles='--')
        plt.hlines(self.lower, 0, self.data.shape[0], linestyles='--')
        plt.plot(self.flag * (np.max(self.energy) - np.min(self.energy)) + np.min(self.energy), '--')
        plt.title(f)
        plt.xlabel('Time Series')
        plt.ylabel('ADC')
        plt.legend(['energy', 'upper limit', 'lower limit', 'touch flag'])
        plt.show()


def make_pos_slice():
    datas = LoadData.load_pos_data('classified data')
    for (data, f) in datas:
        if 'pos4' not in f:
            continue
        dp = DataProcessing(data.reshape([len(data), 1]))
        dp.pre_process()
        dp.calc_moving_avg()
        dp.calc_energy()
        dp.calc_flag()
        dp.show_fig(f)
        dp.calc_valleys(f)


def make_neg_slice():
    len_per_piece = 36
    num_pieces_per_file = 300
    datas = LoadData.load_neg_data('classified data')
    for (data, f) in datas:
        data_len = len(data)
        piece_start = np.random.randint(0, data_len - len_per_piece + 1, [num_pieces_per_file, 1])
        pieces = np.array([data[start[0]:start[0] + len_per_piece] for start in piece_start])
        print(np.shape(pieces))
        pieces = np.apply_along_axis(
            lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)),
            1,
            pieces
        )
        for row in range(pieces.shape[0]):
            np.savetxt(''.join(('data slices/', f[:-4], '_', str(row + 1), '.txt')), pieces[row], fmt='%f')


def make_slice():
    make_pos_slice()
    make_neg_slice()


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'YouYuan'
    plt.rcParams['font.size'] = 20
    make_slice()
