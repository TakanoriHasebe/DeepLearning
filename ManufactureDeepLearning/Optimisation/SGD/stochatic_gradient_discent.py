#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:44:45 2017

@author: Takanori
"""

"""
確率的勾配降下法についてのプログラム

grads : 誤差逆伝播で算出される
"""

# 確率的勾配降下法
class SGD:
    
    # 初期化
    def __init__(self, lr=0.01):
        self.lr = lr

    # 確率的勾配降下法
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]





