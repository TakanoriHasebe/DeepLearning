#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:41:13 2017

@author: Takanori
"""

"""
乗算レイヤと加算レイヤが存在している
"""

# 誤差逆伝搬の加算クラスの実装
class AddLayer:
    
    # 値を保持する必要性はないので, 初期化は行わない
    def __init__(self):
        pass
    
    # 順伝搬の計算
    def forward(self, x, y):
        
        out = x + y
        return out
        
    # 逆伝搬の計算
    def backward(self, dout):
        
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy


# 誤差逆伝搬の乗算クラスの実装
class MulLayer:
    
    # 以下でx, yの初期化を行なっている
    # これらは, 順伝搬時の入力値を保持するために行う
    ## 本当に以下の記述で, 入力値が保存されているかということ ##
    def __init__(self):
        self.x = None
        self.y = None
        
    # 順伝搬の計算
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
        
    # 逆伝搬の計算
    ## 自分のプログラムでは, クラス内の変数から出力値を参照したが, 教科書のプログラムでは出力値を引数で受け取っている ##
    def backward(self, dout):
        dx = dout * self.y # xとyをひっくり返す
        dy = dout * self.x 
        
        return dx, dy



