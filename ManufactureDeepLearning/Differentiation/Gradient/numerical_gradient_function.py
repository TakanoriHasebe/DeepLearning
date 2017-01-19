#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:32:10 2017

@author: Takanori
"""

"""
勾配の実装
"""

import numpy as np

# 関数
def function_2(x):
    
    return x[0]**2 + x[1]**2
    # または return np.sum(x**2)

# 勾配を計算する関数
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状の配列を生成
    
    for idx in range(x.size):
        
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)
         
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
        
    return grad

# 勾配を計算してみる    
print(numerical_gradient(function_2, np.array([3.0, 4.0])))    







