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
# mini_batch_array = np.random.randn(10000, 784)
print(mini_batch_array.shape)
res = zscore(mini_batch_array)
print(res)
print(res.shape)
print(np.mean(res))
print(np.var(res))
# print(res.shape)
