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
class Convolution:
    
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
        col_W = self.W.reshape(FN, -1).T # フィルターを２次元配列に変換
        out = np.dot(col, col_W) + self.b 
                    
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 最終的な出力の形にしている
        
        return out

x = np.array([[[[1,2,3,0], [0,1,2,3], [3,0,1,2], [2,3,0,1]]]])
b = np.array([1])
W = np.array([[[[2,0,1], [0,1,2], [1,0,2]]]])
# x = np.random.randn(1, 1, 2, 2)
# W = np.random.randn(1, 1, 2, 2)
# Convolutionの初期化
conv = Convolution(W, b)
out = conv.forward(x)
print(out)



