#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:38:06 2017

@author: Takanori
"""

"""
P.140の加算レイヤ, 乗算レイヤの図を実装してみる
このプログラムは教科書の図を参考に, 自分自身で実装したものである。

# 課題
特になし
"""

# 誤差逆伝搬の加算クラスの実装
class AddLayer:
    
    # 値を保持する必要性はないので, 各変数の初期化は行わない
    def __init__(self):
        pass
    
    # 順伝搬の計算
    def forward(self, x, y):
        
        out = x + y
        return out
        
    # 逆伝搬の計算
    # 加算クラスでは変わらないから, 1をかけている
    def backward(self, dout):
        
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy


# 誤差逆伝搬の乗算クラスの実装
class MulLayer:
    
    # 以下でx, yの初期化を行なっている
    # これらは, 順伝搬時の入力値を保持するために行う
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

# 各変数の設定
# りんごの個数, りんごの値段, みかんの値段, みかんの個数, 消費税
apple_num = 2
apple = 100
mikan = 150
mikan_num = 3
tax = 1.1

# 値を保持する為に, クラスからインスタンスを作成
# 図をみると, 4層必要であることがわかる
# 3層目の乗算レイヤ
mul_tax_layer = MulLayer()
# 2層目の加算レイヤ
add_price_layer = AddLayer()
# 1層目の上の乗算レイヤ
mul_apple_layer = MulLayer()
# 1層目の下の乗算レイヤ
mul_mikan_layer = MulLayer()

# 順伝搬の実装
# りんごとみかんの各々の値段を算出
apple_price = mul_apple_layer.forward(apple, apple_num)
mikan_price = mul_mikan_layer.forward(mikan, mikan_num)
# りんごとみかんの値段の和を算出
price_sum = add_price_layer.forward(apple_price, mikan_price)
# 合計した値段に対して, 消費税を乗算し, 最終的な値段を算出
price = mul_tax_layer.forward(price_sum, tax)

# 逆伝搬の実装
# 3層目の値段と消費税の変化分の算出
dprice, dtax = mul_tax_layer.backward(1)
print(dprice, dtax)
# 2層目の値段の和の変化分を算出
dapple_price, dmikan_price = add_price_layer.backward(dprice)
print(dapple_price, dmikan_price)
# 1層目の上の乗算レイヤの変化分を算出
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num)
# 1層目の下の乗算レイヤの変化分を算出
dmikan, dmikan_num = mul_mikan_layer.backward(dmikan_price)
print(dmikan, dmikan_num)


