#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:41:34 2017

@author: Takanori
"""

"""
temp
"""

import numpy as np
import math

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)
print('exp_a : ',str(exp_a),'sum_exp_a : ',str(sum_exp_a))
y = exp_a / sum_exp_a
print(y)
