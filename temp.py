#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 11:09:09 2016

@author: Takanori
"""

"""
temp
"""

from sympy import *
import math

x = Symbol('x')

f = 1 / (1 + exp(-x))

grad = diff(f, x)

print(grad.factor())

t = solve(Eq(grad, 0), x)

print(t)

grad0 = diff((1+sin(x))**(0.5), x)

print(grad0)


grad1 = diff(sqrt(1+sin(x)), x)

print(grad1)

if grad0 == grad1:
    
    print('True')


