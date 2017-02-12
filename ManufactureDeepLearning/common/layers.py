#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:59:37 2017

@author: Takanori
"""

"""
ニューラルネットワークで用いられる活性化関数レイヤの
順伝搬, 逆伝搬について記述してある。
基本的にはゼロから作るDeep Learningを参照にプログラムされている。
"""

import numpy as np
from gradientfunctions import numerical_gradient
from outputactivationfunctions import softmax
from lossfunctions import cross_entropy_error
from activatingfunctions import sigmoid

# ReLUレイヤ
class Relu:
    # 初期化
    # mask変数はTrue/FalseからなるNumpy配列であり, 0以下でTrue, 0以上でFalseとなる
    def __init__(self):
        self.mask = None
    
    # 順伝搬    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
           
    # 逆伝搬
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
        
# Sigmoidレイヤ
class Sigmoid:
    
    # 変数の初期化
    def __init__(self):
        self.out = None

    # 順伝搬
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out # 値を保持する
        
        return out
    
    # 逆伝搬
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx

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

# softmax関数と誤差関数のレイヤ
# バッチ学習に対応している
class SoftmaxWithLoss:
    
    # 変数の初期化
    def __init__(self):
        self.loss = None # 誤差
        self.y = None # softmaxの出力
        self.t = None # 教師データ(one-hot-vector)
        
    # 順伝搬
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # print('softmax関数の出力の形状と, 教師データの形状')
        # print(self.y.shape)
        # print(self.t.shape)
        self.loss = cross_entropy_error(self.y, self.t)
        # print(self.loss)
        # print(' ')
        
        return self.loss # 交差エントロピーにより, 誤差が算出された
    
    # 逆伝搬
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        
        # 以下がこのレイヤの誤差逆伝搬になっている
        dx = (self.y - self.t) / batch_size # softmaxの出力と教師データをバッチの個数で割る。
        
        return dx













