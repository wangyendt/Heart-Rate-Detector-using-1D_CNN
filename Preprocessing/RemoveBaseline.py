#!/usr/bin/python
# coding: utf-8
# @Time    : 2018/07/30 17:12
# @Author  : Ye Wang (Wane)
# @Email   : y.wang@newdegreetech.com
# @File    : RemoveBaseline.py
# @Software: PyCharm


import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


class BaselineRemoval:
    def __init__(self):
        self.lam = 1e2  # 1e2 - 1e9
        self.p = 1e-3  # 1e-3 - 1e-1

    @staticmethod
    def baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        z = 0
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def baseline_removing(self, data):
        return np.apply_along_axis(lambda x: self.baseline_als(x, self.lam, self.p), 0, data)


if __name__ == '__main__':
    pass
