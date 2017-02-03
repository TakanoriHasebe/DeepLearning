#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:02:56 2017

@author: Takanori
"""

"""
このプログラムは教科書P.154の図を参照に
自分でsoftmax_with_lossを自分で実装したものである
# 課題
1. 解答では初期化関数でself.loss = Noneで初期化
2. バッチ学習に対応していた
"""

import numpy as np
import sys
sys.path.append('../../')
from common.gradientfunctions import numerical_gradient
from common.outputactivationfunctions import softmax
from common.lossfunctions import cross_entropy_error
from common.activatingfunctions import sigmoid

# softmax_with_loss関数の実装
class SoftmaxLoss:
    
    # 変数の初期化
    def __init__(self):
        self.t = t # 教師データの値を保持
        self.y = y # softmax関数からの出力を保持
        
    # 順伝搬
    # a : ニューラルネットワークからの入力(多次元配列)
    # t : 教師データ(多次元配列)
    def forward(self, a, t):
        
        self.t = t # 教師データの値の保持
        y = softmax(a)
        self.y = y # softmax関数の出力の値の保持
        out = cross_entropy_error(y, t)
        
        return out
    
    # 逆伝搬
    def backward(self):
        
        dout = self.y - self.t
        
        return dout











