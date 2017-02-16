#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:48:34 2017

@author: Takanori
"""

import math
import numpy as np

"""
softmax関数

主に多クラス分類をする際に出力層に用いられる
RBFネットワークでは, 中間層にも用いられている
"""

'''
#オーバーフローの例を以下に記す
a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a))) #オーバーフローしている
c = np.max(a)
print(np.exp(a-c) / np.sum(np.exp(a-c))) #オーバーフローを回避
'''

"""
softmax関数の実装
"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフローを回避
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([1010, 1000, 990])
y = softmax(a)
print(y)
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)






