#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 09:51:15 2017

@author: Takanori
"""

"""
temp
"""

import numpy as np

network = {}
network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
network['W2'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])

a = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
x = np.array([1.0, 0.5]).reshape(1, 2)
print(np.dot(x, a))


print(x.shape)
print(np.dot(x, network['W1']))

a = np.random.rand(3, 4)
print(a.shape)
print(a)


def sigmoid(x): 
    
    return 1 / (1 + np.exp(-x))

 
    
    
    
