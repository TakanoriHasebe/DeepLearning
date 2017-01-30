#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:56:45 2017

@author: Takanori
"""

"""
このプログラムはゼロから作るDeep Learningを元に作成されている
乗算レイヤと加算レイヤに分けられている
"""

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

# りんごの値段, りんごの個数, 消費税
apple = 100
apple_num = 2
tax = 1.1

# layer
# 逆伝搬時の２個目のnodeで用いる
mul_apple_layer = MulLayer()
# 逆伝搬時の１個目のnodeで用いる
mul_tax_layer = MulLayer()

# forward
# りんご２個の計算
apple_price = mul_apple_layer.forward(apple, apple_num)
# りんご２個に対して, 消費税を含めた計算
price = mul_tax_layer.forward(apple_price, tax)
# 最終的な値段の計算
print(price)

# backward
# 最初の誤差逆伝搬の微分の値
# 本来であれば, softmaxなどで誤差が計算される
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
print(dapple_price, dtax)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)





