#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:00:46 2017

@author: Takanori
"""

"""
temp
"""

import numpy as np
import math

# sigmoid関数を多次元配列に対応させるために, numpyの使い方を確かめる
x = np.array([1, 2, 3])

y = 1/(1 + np.exp(-x))
print(y)
