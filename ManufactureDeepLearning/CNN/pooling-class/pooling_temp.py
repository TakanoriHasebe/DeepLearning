#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:06:09 2017

@author: Takanori
"""

"""
poolingのtemp
"""

import numpy as np

arr = np.array([[1,2]])
print('arr:\n'+str(arr))
print(arr.flatten())

print(np.arange(2))

temp = np.zeros((3,3))

temp[np.arange(2), [0,0]] = arr.flatten()

print(temp)