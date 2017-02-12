#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:53:41 2017

@author: Takanori
"""

"""
ニューラルネットで用いられる, 出力関数が記述してある
"""

import numpy as np

# softmax関数
"""
# 以下のコードは自分で作成した
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフローを回避
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y
"""

# softmax関数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))









