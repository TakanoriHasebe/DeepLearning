#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:02:26 2017

@author: Takanori
"""

"""
数値微分の例について
"""

import numpy as np
import matplotlib.pylab as plt

# 関数の微分
def numerical_diff(f, x):
    
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h) # 中心差分（ここにも少しは誤差が含まれている）

# 数値微分の式
def function_1(x):
    
    return 0.01*x**2 + 0.1*x

# グラフに接線を付け加える時に必要
def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

# 数値微分するための式を描画
x = np.arange(0.0, 20.0, 0.1) # 0から20まで, 0.1刻みのx配列
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
#plt.plot(x, y)
#plt.show()

# 実際に微分を計算する
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))

# 微分した際にグラフに接線を付け加える
tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()




