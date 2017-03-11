#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:37:16 2017

@author: Takanori
"""

"""
Batch Normalizationを行うクラスを作成する
* 課題点
1. 計算グラフについて忘れている
2. batch_normalizationについて間違った解釈をしている
   batch_normalizationは強制的にアクティベーションの分布に適度な広がりを持たせる
3. 分散が0になった時の為に, mini_batch_array.std()でなく, np.sqrt(var + 10-9)にしてある
4. 活性化関数に入力するミニバッチ全体の配列に対して, データの標準化をかけるということに注意

参考url
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
"""

import numpy as np
import time
from scipy.stats import zscore

# 教科書のBatchNormalization
class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        
        # 2次元でない場合
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-9)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        
        # ２次元でない場合
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx

print('答えのBatch Normalization')
x = np.array([[1,2,3,4],[5,6,7,8]])
batch_norm = BatchNormalization(1.0, 0.0)
res0 = batch_norm.forward(x)
print(res0)
res = batch_norm.backward(res0)
print(res)




# Batch Normalization
class BatchNormalization:
    
    # 初期化変数
    def __init__(self, gamma, beta, eps):
        
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
        dbeta = np.sum(dout, axis=0) ##
        dgammax = dout
        
        # Step 8
        dgamma = np.sum(dgammax*self.xhat, axis=0)
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
        
        return dgamma, dbeta, dx
print('')
print('自分の作成したBatch Normalizationクラスで確認')
# print(mini_batch_array)
# mini_batch_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
batch_norm = BatchNormalization(np.ones(4), np.zeros(4), 10-9)    
out = batch_norm.forward(x)
print(out)
# print('shape:'+str(out.shape))
# print('mean:'+str(np.mean(out)))
# print('var:'+str(np.var(out)))
dgamma, dbeta, dx = batch_norm.backward(out)
print('dx:'+str(dx))
# print('dgamma:'+str(dgamma))
# print('dbeta:'+str(dbeta))

print('blogのBatch Normalization')
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape

  #step1: calculate mean
  mu = 1./N * np.sum(x, axis = 0)

  #step2: subtract mean vector of every trainings example
  xmu = x - mu

  #step3: following the lower branch - calculation denominator
  sq = xmu ** 2

  #step4: calculate variance
  var = 1./N * np.sum(sq, axis = 0)

  #step5: add eps for numerical stability, then sqrt
  sqrtvar = np.sqrt(var + eps)

  #step6: invert sqrtwar
  ivar = 1./sqrtvar

  #step7: execute normalization
  xhat = xmu * ivar

  #step8: Nor the two transformation steps
  gammax = gamma * xhat

  #step9
  out = gammax + beta

  #store intermediate
  cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache

def batchnorm_backward(dout, cache):

  #unfold the variables stored in cache
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  #get the dimensions of the input/output
  N,D = dout.shape

  #step9
  dbeta = np.sum(dout, axis=0)
  dgammax = dout #not necessary, but more understandable

  #step8
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step7
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar

  #step6
  dsqrtvar = -1. /(sqrtvar**2) * divar

  #step5
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step4
  dsq = 1. /N * np.ones((N,D)) * dvar

  #step3
  dxmu2 = 2 * xmu * dsq

  #step2
  dx1 = (dxmu1 + dxmu2)
  
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)

  #step1
  dx2 = 1. /N * np.ones((N,D)) * dmu
  
  #step0
  dx = dx1 + dx2

  return dx, dgamma, dbeta, dx2

# x = np.array([[1,2,3,4],[5,6,7,8]])
out, cache = batchnorm_forward(x, 1.0, 0.0, 10e-9)
# print(out)
# print(cache)
dx, dgamma, dbeta, dx2 = batchnorm_backward(out, cache)
print('dx:'+str(dx))

