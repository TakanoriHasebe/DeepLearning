#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 08:40:01 2017

@author: Takanori
"""

"""
Batch Normalizationを作成する際のtemp

1. momentumを使用していなくとも, 教科書のBatch Normalizationでは良い結果が得られた
"""

import numpy as np

gamma = 1.0
beta = 0.0
x = np.array([[1,2,3,4],[5,6,7,8]])
input_shape = x.shape
print('教科書のコード')
print('順伝搬')
print('input_shape:'+str(input_shape))
print('x.ndim:'+str(x.ndim))
mu = x.mean(axis=0)
print('mu:'+str(mu))
xc = x - mu
print('xc:'+str(xc))
var = np.mean(xc**2, axis=0)
print('var:'+str(var))
std = np.sqrt(var + 10e-9)
print('std:'+str(std))
xn = xc / std
print('xn:'+str(xn))
out0 = gamma*xn + beta
print('out:'+str(out0))
print('逆伝搬')
dbeta0 = out0.sum(axis=0)
print('dbeta:'+str(dbeta0))
dgamma0 = np.sum(xn * out0, axis=0)
print('dgamma:'+str(dgamma0))
dxn = gamma * out0
print('dxn:'+str(dxn))
dxc = dxn / std
print('dxc:'+str(dxc))
dstd = -np.sum((dxn * xc) / (std * std), axis=0)
print('dstd:'+str(dstd))
dvar = 0.5 * dstd / std
print('dvar:'+str(dvar))
dxc += (2.0 / 2) * xc * dvar
print('dxc:'+str(dxc))
dmu = np.sum(dxc, axis=0)
print('dmu:'+str(dmu))
dx = dxc - dmu / 2
print('dx:'+str(dx))

print('')
print('ブログのコード')
print('順伝搬')
N, D = x.shape
mu = 1./N * np.sum(x, axis = 0)
print('mu:'+str(mu))
xmu = x - mu
print('xmu:'+str(xmu))
sq = xmu ** 2
var = 1./N * np.sum(sq, axis = 0)
print('var:'+str(var))
sqrtvar = np.sqrt(var + 10e-9)
print('std:'+str(sqrtvar))
ivar = 1./sqrtvar
print('ivar:'+str(ivar))
xhat = xmu * ivar
print('xn:'+str(xhat))
out1 = gamma*xhat + beta
print('out:'+str(out1))
print('逆伝搬')
dbeta1 = np.sum(out1, axis=0)
print(dbeta1)
dgammax = out1
dgamma1 = np.sum(dgammax*xhat, axis=0)
print('dgamma:'+str(dgamma1))
dxhat = dgammax * gamma 
print('dxhat:'+str(dxhat))
divar = np.sum(dxhat*xmu, axis=0)
dxmu1 = dxhat * ivar
print('dxmu1:'+str(dxmu1))
dsqrtvar = -1. /(sqrtvar**2) * divar
print('dsqrtvar:'+str(dsqrtvar))
dvar = 0.5 * 1. /np.sqrt(var+10e-100) * dsqrtvar
print('dvar:'+str(dvar))
dsq = 1. /N * np.ones((2,4)) * dvar
dxmu2 = 2 * xmu * dsq
dx1 = (dxmu1 + dxmu2)
print('dx1:'+str(dx1))
dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
print('dmu:'+str(dmu))
dx2 = 1. /N * np.ones((N,D)) * dmu
dx = dx1 + dx2
print('dx:'+str(dx))

