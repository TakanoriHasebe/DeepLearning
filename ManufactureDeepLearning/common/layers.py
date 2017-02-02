#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:59:37 2017

@author: Takanori
"""

"""
ニューラルネットワークで用いられる活性化関数レイヤの
順伝搬, 逆伝搬について記述してある。
基本的にはゼロから作るDeep Learningを参照にプログラムされている。
"""

import numpy as np

# ReLUレイヤ
class Relu:
    # 初期化
    # mask変数はTrue/FalseからなるNumpy配列であり, 0以下でTrue, 0以上でFalseとなる
    def __init__(self):
        self.mask = None
    
    # 順伝搬    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
           
    # 逆伝搬
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
        
# Sigmoidレイヤ
class Sigmoid:
    
    # 変数の初期化
    def __init__(self):
        self.out = None

    # 順伝搬
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out # 値を保持する
        
        return out
    
    # 逆伝搬
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        
        return dx



