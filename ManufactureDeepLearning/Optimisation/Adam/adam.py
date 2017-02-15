#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:34:19 2017

@author: Takanori
"""

"""
勾配を更新する際にAdamを用いる
"""

import numpy as np

class Adam:
    
    # 変数の初期化
    # lr : 学習率, beta1 : 一次モーメント係数, beta2 : 二次モーメント係数
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    # Adamを用いた更新式
    # params : 重み, バイアス行列, grads : 勾配
    def update(self, params, grads):
        
        # 各変数の初期化
        # m : モーメンタム, v : 速度
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)






