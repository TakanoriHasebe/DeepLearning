#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:15:32 2017

@author: Takanori
"""

"""
隠れ層のアクティベーション分布
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(1000, 100)

node_num = 100 # 隠れ層のノード（ニューロン）の数
hidden_layer_size = 5 # 隠れ層が５層
activations = {} # ここにアクティベーションの結果を格納する
              
for i in range(hidden_layer_size):
    
    if i != 0:
        x = activations[i-1]
        
    w = np.random.randn(node_num, node_num) * 1
    
    z = np.dot(x, w)
    a = sigmoid(z) # シグモイド関数
    activations[i] = a

# ヒストグラムの描画
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1)+'-layer')
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()
    










