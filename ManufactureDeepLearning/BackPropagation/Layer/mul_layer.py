#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:20:12 2017

@author: Takanori
"""

"""
誤差逆伝搬法の乗算レイヤの実装
ゼロから作るDeep Learningのりんごの問題に対して, 誤差逆伝搬法を自分で実装したものである

# 結果についての考察
逆伝搬する際に, １つ１つ変数を保存する必要性がある。
以下のコードでは各ノードの入力である, 各arr[0], arr[1]について保存することができていない。
"""

import numpy as np

# 乗算レイヤの実装
# 引数はnumpy形式
# まずは１つのノードに, 2つの入力がある場合について考える
class MulLayer():
    
    # 順伝搬の実装
    def forward(self, arr):
        
        y = arr[0] * arr[1]
        
        return y
    
    # 逆伝搬の実装
    def backforward(self, arr):
        
        pass
    
    
    





