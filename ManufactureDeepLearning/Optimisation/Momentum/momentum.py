#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 14:02:10 2017

@author: Takanori
"""

"""
モーメンタムを用いて最適化する
"""

import numpy as np

class Momentum:
    
    # 変数の初期化
    def __init__(self, lr=0.01, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.v = None # 速度に当たるところ
        
    # 最適化手法の実行
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            # ここでパラメータと同じ配列を作成している
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        # パラメータの更新
        for key in params.keys():
            self.v[key] = self.alpha*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]





