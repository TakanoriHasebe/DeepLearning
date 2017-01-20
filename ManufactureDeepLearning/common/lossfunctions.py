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
    return -np.sum(t * np.log(y + delta)) / batch_size # 平均を求めて正則化s

# 平均２乗誤差
def mean_squard_error(y, t):
    
    return 0.5 * np.sum((y-t)**2)







