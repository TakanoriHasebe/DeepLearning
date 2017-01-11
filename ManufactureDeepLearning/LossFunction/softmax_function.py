#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:47:22 2017

@author: Takanori
"""

"""
softmax関数
"""

import numpy as np

def softmax(a):
    
    exp_a = np.exp(a)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp
    
    return y

a = np.array([0.3, 2.9, 4.0])    
y = softmax(a)
print(y)


