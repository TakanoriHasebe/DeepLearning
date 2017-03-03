#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:48:32 2017

@author: Takanori
"""

"""
データの正規化を行う
平均:0, 分散:1に変換
"""

import numpy as np
from scipy.stats import zscore

mini_batch_array = np.array([1, 2, 3, 4])
# mini_batch_array = np.array([0,0,0,0])
# mini_batch_array = np.random.randn(10000, 784)
print(mini_batch_array.shape)
res = zscore(mini_batch_array)
print(res)
print(res.shape)
print(np.mean(res))
print(np.var(res))
# print(res.shape)
print('')
x = np.array([[1,2,3,4], [5,6,7,8]])
# x_copy = np.copy(x)
# x_std = (x_copy - x_copy.mean()) / x_copy.std()
x_std = (x - x.mean()) / x.std()
print(np.mean(x_std))
print(np.std(x_std))
print('')

x = np.array([[1,2,3,4],[5,6,7,8]])
x_copy = np.copy(x)
x_std = (x_copy - x_copy.mean()) / x_copy.std()
print(np.mean(x_std))
print(np.std(x_std))
print('')

# ミニバッチの配列
# mini_batch_array = np.random.randn(2, 10)
mini_batch_array = np.array([[1,2,3,4], [5,6,7,8]])
# mini_batch_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# mini_batch_array = np.array([[0, 0, 0, 0]])
mean = np.mean(mini_batch_array)
var = np.var(mini_batch_array)
# minus = mini_batch_array - mean
# norm = minus / mini_batch_array.std()
norm = (mini_batch_array - mean) / np.sqrt(var + 10-9)
print(np.mean(norm))
print(np.var(norm))





