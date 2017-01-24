#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 09:47:59 2017

@author: Takanori
"""

"""
ニューラルネットでよく用いられる活性化関数について記述してある。
"""

import numpy as np

# シグモイド関数
def sigmoid(x):
    
    return 1/(1+np.exp(-x))

# relu関数
def relu(x):
    return np.maximum(0, x)



