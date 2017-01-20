#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:53:41 2017

@author: Takanori
"""

"""
ニューラルネットで用いられる, 出力関数が記述してある
"""

import numpy as np

# softmax関数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフローを回避
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y





