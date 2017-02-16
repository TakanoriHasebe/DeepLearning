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
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask

arr = np.random.rand(10)
arr1 = np.random.rand(*arr.shape) # 
print(arr1.shape)
mask = None
mask = arr1 > 0.5
print(arr1)
print(mask)
print(arr1 * mask)


