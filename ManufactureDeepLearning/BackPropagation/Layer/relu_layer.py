#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:24:57 2017

@author: Takanori
"""

"""
活性化関数ReLUの順伝搬, 逆伝搬の実装
このプログラムはP.142の図を参照に自分で実装したものである。
# 課題
以下のプログラムは１つの値のみを受け取るプログラムである。
実際は配列で受け取るので, 配列に対応すべきである
"""

import numpy as np

# relu関数の順伝搬, 逆伝搬の実装
class ReLU:
    
    # 初期化
    # 値を保存しておく為に初期化
    def __init__(self):
        self.x = x
        
    # 順伝搬の実装
    def forward(self, x):
        if x > 0:
            self.x = x
            
            return x
        else:
            self.x = 0
            
            return x

    # 逆伝搬の実装
    def backward(self, dout):
        if dout > 0:
            dx = dout * self.x
            
            return dx
        else:
            dx = dout * 0
            
            return dx

arr = np.array([1, 2, 3])

# インスタンスの作成
relu = ReLU()

# 順伝搬の計算
x = relu.forward(arr)
print(x) # error







