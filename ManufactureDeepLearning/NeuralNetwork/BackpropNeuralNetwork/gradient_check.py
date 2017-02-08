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

import sys,os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3] # 学習データの読み込み
t_batch = t_train[:3] # 教師データの読み込み

print('学習データの形状 : '+str(x_batch.shape))
print('教師データの形状 : '+str(t_batch.shape))
print(' ')

# grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)



"""
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
"""


