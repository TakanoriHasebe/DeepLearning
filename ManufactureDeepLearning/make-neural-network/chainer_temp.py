#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:01:11 2017

@author: Takanori
"""

"""
chainerのテスト
"""

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, \
            Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# 簡単な操作
x1 = Variable(np.array([1], dtype=np.float32))
x2 = Variable(np.array([2], dtype=np.float32))
x3 = Variable(np.array([3], dtype=np.float32))

z = (x1 - 2 * x2 - 1)**2 + (x2 * x3 - 1)**2 + 1
print(z.data)
z.backward()

print(x1.grad)
print(x2.grad)
print(x3.grad)

# softmax_cross_entropy
# 学習データは、2次元のfloat32データ
x = np.array([ [1,2,3,4],[5,6,7,8],[9,10,11,12] ]).astype(np.float32)
# 正解データは、1次元のint32データで0から開始する
t = np.array([0,1,2]).astype(np.int32)
# t = np.array([ [1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0] ]).astype(np.int32)
# print(t.data.shape)

print(x.shape)
print(t.shape)

loss = F.softmax_cross_entropy(x, t)
print(loss.data)

# 正解率
print('正解率')
arr = np.array([[1, 2, 3, 10, 5], [6, 7, 8, 9, 10]])
# print(np.argmax(arr, axis=1))
res = np.argmax(arr, axis=1)
print(res)
print(arr.shape)
t = [3, 4]
accuracy = np.sum(res == t)
print(accuracy / len(t) * 100)
print(res.shape)






