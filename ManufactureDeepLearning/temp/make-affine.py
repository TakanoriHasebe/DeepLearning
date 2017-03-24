#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:12:48 2017

@author: Takanori
"""

"""
Affine
"""

import numpy as np

X = np.random.rand(1, 2)
# print(X.shape)
# print(X)
W = np.random.rand(2, 3)
B = np.random.rand(1, 3)
# print(W.shape)
Y = np.dot(X, W) + B
# print(Y.shape)
print(Y)

class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx

affine = Affine(W, B)
out = affine.forward(X)
print('out:'+str(out))
dout = affine.backward(out)
print('dout:'+str(dout))















