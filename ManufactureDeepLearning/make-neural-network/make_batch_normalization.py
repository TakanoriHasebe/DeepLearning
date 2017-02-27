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

参考url
https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
"""

import numpy as np
import time
from scipy.stats import zscore


# ミニバッチの配列
# mini_batch_array = np.random.randn(2, 10)
# mini_batch_array = np.array([[1, 2, 3, 4]])
mini_batch_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
mini_batch_array = np.array([[1, 2, 3, 4]])
mean = np.mean(mini_batch_array)
var = np.var(mini_batch_array)
minus = mini_batch_array - mean
norm = minus / np.sqrt(var + 10-7)
print(np.mean(norm))
print(np.var(norm))
print('')





"""
start = time.time()
# print('mini_batch_array:'+str(mini_batch_array))
print('mini_batch_array.shape:'+str(mini_batch_array.shape))

# 平均と分散
mean = np.mean(mini_batch_array, axis=1)

var = np.var(mini_batch_array, axis=1)
# print('mean:'+str(mean))
# print('var:'+str(var))

# データの正規化(引き算をするためにfor文を用いる必要がある)
# 10e-7
# mini_batch_norm_test = (mini_batch_array[0] - mean) / np.sqrt(var + 10e-7)
# print(mini_batch_norm_test)
# print((mini_batch_array - mean) / np.sqrt(var + 10e-7))
# mini_batch_array_norm = (mini_batch_array - mean) / np.sqrt(var + 10e-7)
for i in range(len(mini_batch_array)):
    if i==0:
        
        mini_batch_array_norm = (mini_batch_array[i] - mean[i]) / np.sqrt(var[i] + 10e-7)
    else:
        
        mini_batch_array_norm = np.vstack((mini_batch_array_norm, (mini_batch_array[i] - mean[i]) / np.sqrt(var[i] + 10e-7)))
# print('mini_batch_array_norm:'+str(mini_batch_array_norm))
print('mini_batch_array_norm.mean:'+str(np.mean(mini_batch_array_norm)))
print('mini_batch_array_norm.var:'+str(np.var(mini_batch_array_norm)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))

print('')
print('axis=0')
print('mini_batch_array.shape:'+str(mini_batch_array.shape))
start = time.time()

# 平均と分散
mean = np.mean(mini_batch_array, axis=0)
var = np.var(mini_batch_array, axis=0)

# 正規化
mini_batch_array_norm = (mini_batch_array - mean) / np.sqrt(var + 10e+7)
# print('mini_batch_norm_array:'+str(mini_batch_norm_array))
print('mini_batch_array_norm.mean:'+str(np.mean(mini_batch_array_norm)))
print('mini_batch_array_norm.var:'+str(np.var(mini_batch_array_norm)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))
"""
"""

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
"""




