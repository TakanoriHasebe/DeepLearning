#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:49:51 2017

@author: Takanori
"""

"""
過学習を抑制するdropout手法について
"""

import numpy as np
"""
X = np.random.randn(2)
W0 = np.random.randn(2,2)
W1 = np.random.randn(2)

z = np.dot(X,W0)
y = np.dot(z,W1)

print(y)
"""

X = np.array([1,2])
W0 = np.array([[1,2],[3,4]])
W1 = np.array([2,1])

z = np.dot(X,W0)
print(z)
W1[0] = 0
y = np.dot(z,W1)
print(y)

