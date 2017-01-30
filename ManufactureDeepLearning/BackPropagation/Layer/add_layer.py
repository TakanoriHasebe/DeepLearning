#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:26:20 2017

@author: Takanori
"""

"""
誤差逆伝搬法の加算レイヤの実装
ゼロから作るDeep Learningの加算レイヤを自分で実装したもの

# 結果に対する考察
加算レイヤでは値を保持する必要がないので, 初期化をする必要性はない
また, ２値入力であるから逆伝搬時も２つの値が必要になる
1. __init__の部分をpassに
2. これを考えるとforwardの所の計算を変わってくる
3. ２つの値が必要であるから, dx, dyを返すようにする
"""

# 誤差逆伝搬時に用いられる, 加算レイヤの実装
class AddLayer:
    
    # 2値入力に対して, 値を保持する
    def __init__(self):
        self.x = None
        self.y = None
    
    # 順伝搬
    # 順伝搬では入力された値の和をとるので, x + y
    def forward(self, x, y):
        self.x = x
        self.y = y
        add = x + y
        
        return add
        
    # 逆伝搬
    # そのまま値を逆伝搬するので, return dout
    def backward(self, dout):
        return dout
        
        
        
        
        
        


