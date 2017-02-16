#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:17:00 2017

@author: Takanori
"""

"""
temp
"""

import numpy as np

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.array([[0.1, 0.8, 1.0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
arr = np.array([1, 2, 3])

print(x)

print(np.argmax(x, axis=1))
print(np.argmax(y, axis=1))

p = np.argmax(x, axis=1)
t = np.argmax(y, axis=1)

print(np.sum(p == t))






