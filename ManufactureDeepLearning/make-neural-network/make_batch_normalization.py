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

# ミニバッチの配列
# mini_batch_array = np.random.randn(2, 3)
print('順伝搬の計算')
mini_batch_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
out, cache = batchnorm_forward(mini_batch_array, 1.0, 0.0, 10-9)
print('out.shape:'+str(out.shape))
print(np.mean(out))
print(np.var(out))
print(cache)


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

  return dx, dgamma, dbeta
print('')
print('逆伝搬の計算')
dx, dgamma, dbeta = batchnorm_backward(out, cache)
print(dx)
print(dgamma)
print(dbeta)

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
        
        # backward
        

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

print('自分の作成したBatch Normalizationクラスで確認')
# print(mini_batch_array)

batch_norm = BatchNormalization(1.0, 0.0, 10-9)    
out = batch_norm.forward(mini_batch_array)
print('shape:'+str(out.shape))
print('mean:'+str(np.mean(out)))
print('var:'+str(np.var(out)))

dgamma, dbeta, dx = batch_norm.backward(out)
print(dx)
print(dgamma)
print(dbeta)



