#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:09:13 2017

@author: Takanori
"""

"""
pooling層の実装
"""

import numpy as np

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    

    return img[:, :, pad:H + pad, pad:W + pad]

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

# プーリング層
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        # print('pool_size:'+str(pool_size))
        
        '''''''''''''''''''''''''''
        最終的な配列（入力配列）の型を作成
        '''''''''''''''''''''''''''
        dmax = np.zeros((dout.size, pool_size))
        # print('dmax.shape:'+str(dmax.shape))
        # print(dout.flatten())
        
        '''''''''''''''''''''''''''
        0を入れることでpaddingをしている
        '''''''''''''''''''''''''''
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        # print('')
        # print('dmax\n:'+str(dmax))
        # print('arg_max:\n'+str(self.arg_max))
        # print('arg_max.size:'+str(self.arg_max.size))
        
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        # print('dmax:\n:'+str(dmax))
        # print('dmax.shape:\n'+str(dmax.shape))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        # print('dcol:\n'+str(dcol))
        
        '''''''''''''''''''''''''''
        普通の配列からimageに変換
        '''''''''''''''''''''''''''
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        # print('dx:\n'+str(dx))
        
        return dx


x = np.array([[[[1,2,3,0], [0,1,2,3], [3,0,1,2], [2,3,0,1]]]])
x = np.array([[[[1,1],[1,1],[1,1],[1,1]]]])
# print('x:\n'+str(x))
b = np.array([1])
W = np.array([[[ [1,1],[1,1] ]]])
pool = Pooling(2,2)
out = pool.forward(x)
# print('')
# print('out:\n'+str(out))
dout = pool.backward(out)
print('')
print('dout:\n'+str(dout))

