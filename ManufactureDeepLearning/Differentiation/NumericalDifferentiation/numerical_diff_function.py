#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 10:48:59 2017

@author: Takanori
"""

"""
丸め誤差と微分の誤差を改善をしたもの

中心差分も数値微分であるので, 誤差が含まれている。
"""

import numpy as np

# 関数の微分
def numerical_diff(f, x):
    
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h) # 中心差分（ここにも少しは誤差が含まれている）





