#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:27:22 2017

@author: Takanori
"""

"""
2層ニューラルネットのクラス
"""

import sys
sys.path.append('../../')

from common.gradientfunctions import numerical_gradient
from common.outputactivationfunctions import softmax
from common.lossfunctions import cross_entropy_error
from common.activatingfunctions import sigmoid
import numpy as np
from mnist import load_mnist


# 2層ニューラルネットのクラス
class TwoLayerNet:
    
    # 初期化関数
    # 引数は, 入力層のニューロンの数, 隠れ層のニューロンの数, 出力層のニューロンの数
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * \
                  np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                   np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    # 認識を行う関数
    # 引数は, 画像データ
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = sigmoid(a2)
        
        return y
    
    # 損失関数の値を求める関数
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
        
    # 認識制度を求める関数
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 重みパラメータに対する勾配を求める
    # x:入力データ, t:教師データ
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t) 
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads

"""
print('初期状態の重み, バイアスの形状をみる')        
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)        

print(' ')
# 推論処理は以下のよう
# 入力からどのような出力を得るかということを出力する。
print('推論処理について')
x = np.random.rand(100, 784) # ダミーの入力データ(100枚分)
y = net.predict(x) # 推論       
print(y.shape) # ダミーの入力データを100枚入れたので, 100枚分の推論が出力される
print(' ')
# 勾配処理は以下のよう
# 入力データと教師データを与え, どのような勾配が算出されるかをみる。
x = np.random.rand(100, 784) # ダミーの入力データ(100枚分)
t = np.random.rand(100, 10) # ダミーの正解ラベル(100枚分)

grads = net.numerical_gradient(x, t) # 勾配を計算
print('勾配処理について')
print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)                              
"""





