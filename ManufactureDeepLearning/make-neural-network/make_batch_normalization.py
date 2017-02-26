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


# ミニバッチの配列
mini_batch_array = np.random.randn(5, 10)
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
# ミニバッチの配列
# print('mini_batch_array.shape:'+str(mini_batch_array.shape))
start = time.time()
# 平均と分散
mean = np.mean(mini_batch_array, axis=0)
var = np.var(mini_batch_array, axis=0)
# print('mean:'+str(mean))
# print('var:'+str(var))

# 正規化
mini_batch_array_norm = (mini_batch_array - mean) / np.sqrt(var + 10e+7)
# print('mini_batch_norm_array:'+str(mini_batch_norm_array))
print('mini_batch_array_norm.mean:'+str(np.mean(mini_batch_array_norm)))
print('mini_batch_array_norm.var:'+str(np.var(mini_batch_array_norm)))
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time))
# print('mini_batch_array_norm:'+str(mini_batch_array_norm))
# print('mini_batch_array_norm.mean:'+str(np.mean(mini_batch_array_norm)))
# print('mini_batch_array_norm.var:'+str(np.var(mini_batch_array_norm)))






"""
# 正規化データに変換をかける
# gamma, beta : パラメータ
gamma = 1
beta = 0
# print(mini_batch_array * 2)
mini_batch_array_norm_conversion = mini_batch_array_norm * gamma + beta
print('mini_batch_array_norm_conversion:'+str(mini_batch_array_norm_conversion))

# Batch Normalizationの順伝搬, 逆伝搬のクラスを作成
class BatchNormalization:
    
    # 変数の初期化
    def __init__(self):
        self.gamma = 1
        self.beta = 0
        

    # 順伝搬
    def forward(self, mini_batch_array):
        mean = np.mean(mini_batch_array) # 平均
        var = np.var(mini_batch_array) # 分散
        mini_batch_array_norm = (mini_batch_array - mean) / np.sqrt(var + 10e-7) # ミニバッチの正規化
        mini_batch_array_norm_conversion = mini_batch_array_norm * self.gamma + self.beta # スケールとシフト
        out = mini_batch_array_norm_conversion # 最終結果
        
        return out   
    
    # 逆伝搬
    def backward(self):
        
        pass


# batch_normalization = BatchNormalization()
# res = batch_normalization.forward(mini_batch_array)
# print('res:'+str(res))
"""





