#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:23:17 2017

@author: Takanori
"""

"""
3層ニューラルネットワークの実装

"""

import numpy as np

"""
sigmoid function
sigmoid関数
"""
def sigmoid(x): 
    
    return 1 / (1 + np.exp(-x))

"""
network's weight and bias
ネットワークの重み, バイアスの設定をしている    
"""    
def init_network():
    
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

"""
forward transfer function
順伝搬関数の作成
"""    
def forward(network, x):
    
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    print('a1: '+str(a1))
    z1 = sigmoid(a1)
    print('z1: '+str(z1))
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    print('z2: '+str(z2))
    a3 = np.dot(z2, W3) + b3
    print('a3: '+str(a3))
    y = a3 #恒等写像 identity_function(a3)
    
    return y
    
network = init_network() #辞書形式で重み, バイアスが作成された
x = np.array([1.0, 0.5]).reshape(1, 2) #入力
print('x: '+str(x))
y = forward(network, x) #順方向の計算
print(y)

    
    
    