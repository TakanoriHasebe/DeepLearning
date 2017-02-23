#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:07:33 2017

@author: Takanori
"""

"""
ニューラルネットワークを作成する際のtempプログラム
"""

import sys, os
sys.path.append('../common')  # 親ディレクトリのファイルをインポートするための設定
from layers import Sigmoid, Affine, SoftmaxWithLoss # 誤差逆伝搬の時に必要
from collections import OrderedDict
import numpy as np

# 最初の重みの設定
input_size = 3
hidden_size = 2

print(np.array([input_size, hidden_size]))
# print(np.random.randn(input_size, hidden_size))
# print(np.random.randn(hidden_size))
arr0 = np.random.randn(input_size, hidden_size)
arr1 = np.random.randn(hidden_size)
# print(arr0)
# print(arr1)
# print(arr0 + arr1)

# numpyの足し算
arr0 = np.array([[1, 2, 3], [4, 5, 6]])
arr1 = np.array([1, 2, 3])
print(arr0 + arr1)

# OrderedDictについて
d = {}
d['a'] = np.random.randn(2, 3)
d['c'] = np.random.randn(3, 4)
print(d)

d = OrderedDict()
d['a'] = np.random.randn(2, 3)
d['c'] = np.random.randn(3, 4)
print(d)

for t in d.values():
    
    print(t)

# 他のクラスを呼び出す
# 辞書内にクラスのインスタンスを作成する
# Affineレイヤ
# バッチ学習に対応している
class Affine:
    
    # 保存する各変数の初期化
    def __init__(self, W, b):
        self.W = W # 重み
        self.b = b # バイアス
        self.x = None # 入力
        self.dW = None # 重みの微分 
        self.db = None # バイアスの微分
        
    # 順伝搬
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
                    
        return out
                    
    # 逆伝搬
    def backward(self, dout):
        # print('dout : '+str(dout))
        # print(self.W.T.shape)
        # print('self.W.T : '+str(self.W.T))
        dx = np.dot(dout, self.W.T) # 入力の逆伝搬
        self.dW = np.dot(self.x.T, dout) # 重みの逆伝搬
        self.db = np.sum(dout, axis=0) # バイアスの逆伝搬
        
        return dx

params = {}
x = np.random.randn(100, 100)
params['W1'] = np.random.randn(100, 50)
params['b1'] = np.random.randn(50)
params['W2'] = np.random.randn(50, 10)
params['b2'] = np.random.randn(10)
layers = OrderedDict()
layers['Affine1'] = Affine(params['W1'], params['b1'])
layers['Affine2'] = Affine(params['W2'], params['b2'])
for layer in layers.values():
    x = layer.forward(x)
    
print(x.shape)

# 逆伝搬で用いるためにオブジェクトを逆にする
layers = list(layers.values())
print(layers)
layers.reverse()
print(layers)

# バッチ学習
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
size = 2
batch_mask = np.random.choice(arr.shape[0], size)
print(batch_mask)
arr0 = arr[batch_mask]
print(arr0)

# iter_per_epochについて
iter_per_epoch = max(60 / 20, 1)
print(iter_per_epoch)
iter_per_epoch = max(0 / 20, 1)
print(iter_per_epoch)

# if文の確認
current_iter = 0
iter_per_epoch = max(60 / 20, 1)
if current_iter % iter_per_epoch == 0:

    print('True')

# if文の確認（None）
evaluate_sample_num_per_epoch = None
if not None is None:
    
    print('None')

# argmaxについて
y = np.array([0.01, 0.2, 0.4, 1])
y = np.argmax(y, axis=0)
print(y)
















