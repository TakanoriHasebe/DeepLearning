#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 19:07:19 2017

@author: Takanori
"""

"""
ニューラルネットワークを例にして, 勾配を求めるプログラム
"""

import sys
sys.path.append('../../')
import numpy as np
from common.gradientfunctions import numerical_gradient
from common.outputactivationfunctions import softmax
from common.lossfunctions import cross_entropy_error

# 簡単なニューラルネットで勾配を算出する
class simpleNet:
    
    def __init__(self):
        self.W = np.random.randn(2, 3) # 重みをガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)
    
    # x:入力データ, t:正解ラベル
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
   
net = simpleNet()
print('初期重みパラメータ')
print(str(net.W)) # 重みパラメータ
print(' ')
x = np.array([0.6, 0.9])
p = net.predict(x)
print('入力と重みの内積')
print(p)
print(' ')
print('最大値のインデックス')
print(np.argmax(p)) # 最大値のインデックス
t = np.array([0, 0, 1]) # 正解ラベル
print(' ')
print('交差エントロピーによる誤差の算出')
print(net.loss(x, t))
     
# 関数を設定
def f(W):
    return net.loss(x, t)

dW = numerical_gradient(f, net.W)
print(' ')
print('重みの微分（１つ１つ偏微分している）')
print(dW)














