#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:08:04 2017

@author: Takanori
"""

"""
3層ニューラルネットワーク
3入力で2出力
"""

import numpy as np

"""
sigmoid関数
"""
def sigmoid(x): 
    
    return 1 / (1 + np.exp(-x))

"""
ネットワークの重み, バイアスの設定
"""
def init_network():
    
    network = {} #辞書形式
    
    #1層目の重み, バイアスの設定
    network['W1'] = np.random.rand(3, 4) 
    network['b1'] = np.random.rand(1, 4)
    
    #2層目の重み, バイアスの設定
    network['W2'] = np.random.rand(4, 3)
    network['b2'] = np.random.rand(1, 3)
    
    #3層目の重み, バイアスの設定
    network['W3'] = np.random.rand(3, 2)
    network['b3'] = np.random.rand(1, 2)
    
    return network
    

"""
順伝搬関数
"""
def forward(x, network):
    
    #1層目の計算
    W1 = network['W1']
    b1 = network['b1']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    #2層目の計算
    W2 = network['W2']
    b2 = network['b2']
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    #3層目の計算
    W3 = network['W3']
    b3 = network['b3']
    a3 = np.dot(z2, W3) + b3
    
    return a3
    
    

network = init_network() #networkの作成    
x = np.array([0.1, 0.2, 0.3]).reshape(1, 3) #入力
y = forward(x, network) #順伝搬関数に代入し出力を得る

print('入力の次元数 : '+str(x.shape[1]))
print('出力の次元数 : '+str(y.shape[1]))
print(' ')
print('入力 : '+str(x))
print('出力 : '+str(y))








