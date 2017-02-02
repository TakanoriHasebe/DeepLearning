#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:31:56 2017

@author: Takanori
"""

"""
Affineレイヤ
このプログラムではAffineレイヤの例が記述してある
"""

import numpy as np

# 以下では一般的なAffine変換の簡単な流れが記述されている
X = np.random.rand(2) # 入力
W = np.random.rand(2, 3) # 重み
B = np.random.rand(3) # バイアス

Y = np.dot(X, W) + B #2次元から3次元に変換された
 
# 以下ではバッチ版Affine変換の簡単な流れが記述されている
# ここで入力のバッチサイズを2個とする
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])          
          
print(X_dot_W)
print(X_dot_W + B)
print(' ')          
dY = np.array([[1, 2, 3], [4, 5, 6]]) 
print(dY)         
dB = np.sum(dY, axis=0)          
print(dB)          
          








          