#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:57:25 2017

@author: Takanori
"""

"""
丸め誤差と前方差分
"""
import numpy as np


# 関数の微分
def numerical_diff(f, x):
    h = 10e-50 # 丸め誤差
    return (f(x+h) - f(x)) / h # 前方差分

# 丸め誤差の例
print(np.float32(10e-50)) # あまりにも数値が小さくて, 0.0と表示される

print(np.float32(10e-4)) # こちらを用いると良い結果を得られる

     
     
     
     
     
     
     
     