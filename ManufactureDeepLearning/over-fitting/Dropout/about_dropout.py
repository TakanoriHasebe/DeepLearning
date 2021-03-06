#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:39:16 2017

@author: Takanori
"""

"""
dropoutについてのプログラム
"""

import numpy as np

# Dropoutについてのプログラム
class Dropout:
    
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        
    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # print(self.mask)
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask

x = np.array([1,2,3,4,5])
drop = Dropout()
res = drop.forward(x)
print(res)
res = drop.backward(res)








