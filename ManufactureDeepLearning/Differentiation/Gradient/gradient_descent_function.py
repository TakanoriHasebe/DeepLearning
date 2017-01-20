#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:52:59 2017

@author: Takanori
"""

"""
勾配法
"""

import numpy as np

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


# 勾配法の実装
# f:最適化したい関数, lr:学習率, step_num:繰り返しの数
def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return x
    
# 勾配を算出したい関数
def function_2(x):
    
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))










