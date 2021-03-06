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
# from gradientfunctions import numerical_gradient
from outputactivationfunctions import softmax
from lossfunctions import cross_entropy_error
# from functions import sigmoid, softmax, cross_entropy_error
# from activatingfunctions import sigmoid

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
        # print(dout.shape, self.W.shape)
        dx = np.dot(dout, self.W.T) # 入力の逆伝搬
        self.dW = np.dot(self.x.T, dout) # 重みの逆伝搬
        self.db = np.sum(dout, axis=0) # バイアスの逆伝搬
        
        return dx


# softmax関数と誤差関数（cross_entropy_error）のレイヤ
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
    
    """
    # 逆伝搬
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        
        # 以下がこのレイヤの誤差逆伝搬になっている
        dx = (self.y - self.t) / batch_size # softmaxの出力と教師データをバッチの個数で割る。
        
        return dx
    """
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

'''
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
'''

# Batch Normalization
class BatchNormalization:
    
    # 初期化変数
    def __init__(self, gamma, beta, eps = 10e-9):
        
        # 変数群
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        
        # forward
        self.mu = None # 平均
        self.xmu = None # ミニバッチから平均を減算
        self.sq = None # 分散を求めるために必要
        self.var = None # 分散
        self.sqrtvar = None # 標準偏差
        self.ivar = None # 標準偏差の反転
        self.xhat = None # Normalizationした配列
        
        # backward
        self.dbeta = None
        self.dgamma = None

    # 順伝搬
    def forward(self, x):
        
        N, D = x.shape
        
        # 平均の算出
        self.mu = 1. / N * np.sum(x, axis=0)
        
        # ミニバッチから平均を引く
        self.xmu = x - self.mu
        
        # 分散を計算するのに必要
        self.sq = self.xmu ** 2
        
        # 分散の算出
        self.var = 1. / N * np.sum(self.sq, axis=0)
        
        # 標準偏差の計算
        self.sqrtvar  = np.sqrt(self.var + self.eps)
        
        # 標準偏差の反転
        self.ivar = 1. / self.sqrtvar
        
        # Normalizationの実行
        self.xhat = self.xmu * self.ivar
        
        # gammaxの計算
        gammax = self.gamma * self.xhat
        
        # 最終結果の計算
        out = gammax + self.beta
        
        return out
    
    # 逆伝搬
    def backward(self, dout):
        
        N, D = dout.shape
        
        # Step 9
        self.dbeta = np.sum(dout, axis=0) ##
        dgammax = dout
        
        # Step 8
        self.dgamma = np.sum(dgammax*self.xhat, axis=0)
        dxhat = dgammax * self.gamma
        
        # Step 7
        divar = np.sum(dxhat*self.xmu, axis=0)
        dxmu1 = dxhat*self.ivar
        
        # Step 6
        dsqrtvar = -1./(self.sqrtvar**2)*divar
        
        # Step 5
        dvar = 0.5 * 1./np.sqrt(self.var+self.eps) * dsqrtvar
        
        # Step 4
        dsq = 1./N*np.ones((N, D)) * dvar
                          
        # Step 3
        dxmu2 = 2*self.xmu*dsq
        
        # Step 2
        dx1 = (dxmu1+dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        
        # Step 1
        dx2 = 1./N*np.ones((N, D))*dmu
                          
        # Step 0
        dx = dx1 + dx2
        
        return dx




