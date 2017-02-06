#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:28:23 2017

@author: Takanori
"""

"""
このプログラムは教科書P.161を参照に書かれた。
数値微分の匂配と, 誤差逆伝搬法で求めた匂配が等しいかをチェックする。
"""

import sys
sys.path.append('../MNIST')
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 入力:784, 隠れ層:50, 出力:10
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))




