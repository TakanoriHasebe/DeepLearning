#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:00:02 2017

@author: Takanori
"""

"""
交差エントロピー関数
"""

import numpy as np

# ミニバッチ学習に対応した交差エントロピー関数
def cross_entropy_error(y, t):
    
    # ラベル入力の場合, one-hot-vectorに変換
    if t.ndim == 0:
        arr = np.zeros(10, int) # ここで作成するベクトルは, 問題に応じて変更する必要性がある
        arr[t] = 1
        t = arr
    
    # 入力が１つの場合
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    
    delta = 1e-7 # ここを注意する
    batch_size = y.shape[0] # ミニバッチのサイズ
    return -np.sum(t * np.log(y + delta)) / batch_size # 平均を求めて正則化

t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]
y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]]

print(cross_entropy_error(np.array(y), np.array(t)))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print(cross_entropy_error(np.array(y), np.array(t)))





