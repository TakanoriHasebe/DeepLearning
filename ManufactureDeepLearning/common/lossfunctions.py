#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:48:00 2017

@author: Takanori
"""

"""
ニューラルネットで用いられる, 誤差関数が記述してある
"""

import numpy as np

# ミニバッチ学習に対応した交差エントロピー関数
def cross_entropy_error(y, t):
    
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

    '''
    # 以下のコードは自分で実装したものだが, エラーがある
    # ラベル入力の場合, one-hot-vectorに変換
    if t.ndim == 0:
        arr = np.zeros(10, int) # ここで作成するベクトルは, 問題に応じて変更する必要性がある
        arr[t] = 1
        t = arr
    
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    # 入力が１つの場合
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    
    delta = 1e-7 # ここを注意する
    batch_size = y.shape[0] # ミニバッチのサイズ
    # print('batch_size : '+str(batch_size))
    return -np.sum(t * np.log(y + delta)) / batch_size # 平均を求めて正則化
    '''
    
    
# 平均２乗誤差
def mean_squard_error(y, t):
    
    return 0.5 * np.sum((y-t)**2)







