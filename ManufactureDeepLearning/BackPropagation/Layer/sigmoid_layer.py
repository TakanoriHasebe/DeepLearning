#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:53:15 2017

@author: Takanori
"""

"""
このプログラムは
ゼロから作るDeep Learningの誤差逆伝搬法の
sigmoid関数の部分を自分で実装したものである。
# 課題

"""

import numpy as np

#  sigmoid関数の実装
class Sigmoid:
    
    # 変数の初期化
    # 計算グラフより変数xは保持しなくても良い
    def __init__(self):
        self.y = None
        
    # 順伝搬
    # 多次元配列で受け取るということに注意する
    def forward(self, x, y):
        pass



