#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:40:28 2017

@author: Takanori
"""

"""
image to columns関数
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
    print('N:'+str(N)+' C:'+str(C)+' H:'+str(H)+' W'+str(W))
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    print('out_h:'+str(out_h), 'out_w:'+str(out_w))

    # ここでpaddingを施す
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    print('img.shape:'+str(img.shape))
    
    # 初期化
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    print('col.shape:'+str(col.shape))

    
    for y in range(filter_h): # 5
        y_max = y + stride*out_h
        # print(y_max)
        # print('y_max:'+str(y_max))
        for x in range(filter_w): # 5
            x_max = x + stride*out_w
            # print(x_max)
            # print('x_max:'+str(x_max))
            print(col)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # print(col)
            # print(col.shape)
            
    print('col.shape:'+str(col.shape))
    col_temp = col.transpose(0, 4, 5, 1, 2, 3)
    print('col.shape:'+str(col.shape))
    print('N*out_h*out_w:'+str(N*out_h*out_w))
    col = col_temp.reshape(N*out_h*out_w, -1) # 行数の指定
    return col

# バッチサイズ, チャンネル, 画像サイズ
x1 = np.random.rand(1, 1, 2, 2)
# print(x1[0][0][0])
# print(x1[0][0][1])
# print(x1[0][0].shape)
# print(x1[0])
print('')
# フィルターの高さと幅
col1 = im2col(x1, 2, 2, stride=1, pad=0)
print('col1.shape:'+str(col1.shape))




