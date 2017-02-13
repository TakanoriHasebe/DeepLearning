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

# 辞書
print(' ')
d = {}
d['A1'] = np.array([1,2,3])
d['A2'] = np.array([1,2,4])
print(d)

temp = list(d.values())
print(temp)
temp.reverse()
print('逆順にアクセスする')
for i in temp:
    
    print(i)
    
# クラス内の変数にアクセス
print(' ')
print('クラス内の変数にアクセス')
class Temp():
    
    def __init__(self):
        self.a = None

    def b(self, n):
        
        self.a = n
        

t = Temp()
t.b(5)
print(t.a)

# 辞書
d = {}
d['W1'] = np.array([1,2,3])
d['W2'] = np.array([1,2,3,4])
v = {}

# itemsでどちらも表示
for key, val in d.items():
    print(key, val)
    v[key] = np.zeros_like(val)







