#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:17:15 2017

@author: Takanori
"""

"""
畳み込み層の実装
im2colを用いる
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

# 畳み込み層の実装
class make_Convolution:
    
    # 初期化関数    
    # フィルター, バイアス, ストライド, padding
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    # forward
    def forward(self, x):
        FN, C, FH, FW = self.W.shape # FN:フィルターの個数, C:チャネル, FH:フィルターの高さ, FW:フィルターの幅
        N, C, H, W = x.shape 
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + W*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad) # 入力データを２次元配列に変換
        print('col.shape:'+str(col.shape))
        col_W_temp = self.W.reshape(FN, -1) # フィルターを２次元配列に変換
        print('col_W_temp.shape:'+str(col_W_temp.shape))
        col_W = col_W_temp.T
                                   
        out = np.dot(col, col_W) + self.b
        print('out.shape:'+str(out.shape))
                    
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 最終的な出力の形にしている
        
        return out
    
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        
        print('col_W.shape:'+str(self.col_W.shape))
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0) 
        self.dW = np.dot(self.col.T, dout) 
        # print('dW.shape:'+str(self.dW.shape))
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        # print('self.dW.shape:'+str(self.dW.shape))

        dcol = np.dot(dout, self.col_W.T)
        # print('dcol.shape:'+str(dcol.shape))
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

x = np.array([[[[1,2,3,0], [0,1,2,3], [3,0,1,2], [2,3,0,1]]]])
b = np.array([1])
W = np.array([[[ [1,1],[1,1] ]]])
# x = np.random.randn(1, 1, 2, 2)
# W = np.random.randn(1, 1, 2, 2)
# Convolutionの初期化
conv = Convolution(W, b)
out = conv.forward(x)
print(out.shape) 
dout = conv.backward(out)




