#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:53:48 2017

@author: Takanori
"""

"""
２乗誤差
"""

import numpy as np

"""
平均２乗誤差関数
"""
def mean_squard_error(y, t):
    
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(mean_squard_error(np.array(y), np.array(t))) #2番目が1番大きい数値

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(mean_squard_error(np.array(y), np.array(t))) #7番目が1番大きい数値


