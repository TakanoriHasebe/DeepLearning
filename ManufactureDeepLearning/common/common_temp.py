#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:55:33 2017

@author: Takanori
"""

"""
ニューラルネットワークで一般的に用いられる関数を試している
"""

from lossfunctions import cross_entropy_error
import numpy as np

# 勾配の確認の交差エントロピーのところでエラーが発生したので, 修正した。
loss = cross_entropy_error(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 1]))
print(loss)
print(-np.log(0.3))

# pythonのクラスのtemp
class Test:
    
    def __init__(self):
        
        self.x = None
        
    def a(self, a):
        
        self.x = a
        
        return a

test = Test()
a = test.a(1)


# list
a = [1, 2, 3, 4]
print(a)
b = a.reverse()
print(b)




