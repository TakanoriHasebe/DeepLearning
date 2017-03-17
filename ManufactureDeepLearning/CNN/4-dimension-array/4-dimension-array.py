#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:15:09 2017

@author: Takanori
"""

"""
4次元配列
"""

import numpy as np

# 4-dimension
x = np.random.rand(10, 1, 28, 28)

y = np.random.rand(2, 2, 2, 1)

y0 = np.array([[[1,2,3,4,5,6,7]]])
y1 = np.array([[[[1,2], [1,2], [1,2], [1,2]]]])
print(y0.shape)
print(y1.shape)

# pad
arr = np.array([[[[1,2],[1,2]]]])
print(arr)
print(arr.shape)

img = np.pad(arr, [(0,0), (0,0), (1,1), (1,1)], 'constant')
print(img)
print(img.shape)

# array
arr = np.array([[1,2],[3,4]])
print(arr)

temp = arr[0][0:2:1]
print(temp)

# transpose
arr = arr.transpose(1, 0)
print(arr)
arr = np.array([[1,2,3], [4,5,6]])
print(arr.transpose(1,0))

# reshape
arr = np.array([[1,2,3,4,5,6,7,8]])
arr = arr.reshape(4, -1) # 必ず１行になる
print(arr)

# reshape
arr = np.array([1])
out = arr.reshape(1,1,1,1).transpose(0, 3, 1, 2)
print(out)



