#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 11:27:06 2017

@author: Takanori
"""

"""
このプログラムは教科書P.158を参照に記述された
誤差逆伝搬法をレイヤを用いて実装している
"""

import sys
# sys.path.append('/Users/Takanori/Desktop/AI/DeepLearning/ManufactureDeepLearning/')
sys.path.append('../../common')
import numpy as np
from common.gradientfunctions import numerical_gradient
from common.outputactivationfunctions import softmax
from common.lossfunctions import cross_entropy_error
from common.layers import Relu, Sigmoid, Affine, SoftmaxWithLoss
from collections import OrderedDict

class TwoLayerNet:

    # 引数：入力ニューロンの数, 隠れ層ニューロンの数, 出力層ニューロンの数, 重み初期化時のガウス分布のスケール
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        
        # 重みの初期化（辞書形式）
        # 重みはランダムで初期化している。バイアスは０で初期化している。
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # レイヤの生成（辞書形式）
        # 作成したAffineレイヤ, ReLUレイヤがそれぞれの内部で順伝搬と逆伝搬を正しく処理するので, 以下ではレイヤを正しい順番で連結し, 呼び出す
        self.layers = OrderedDict() # 追加した順にレイヤのforward()メソッドを呼び出すためにOrderedDict()を利用
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
    
    
    # softmax関数と誤差関数までのニューラルネットワーク
    def predict(self, x):
        for layer in self.layers.values():
            # print(layer)
            # print(x.shape)
            x = layer.forward(x)
            # print(x.shape)
            # print(x[0]) # 最初のデータをみる
            # print(' ')
            
        return x
   
    
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x) 
        return self.lastLayer.forward(y, t) 
    
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy    
    
    # 誤差逆伝搬
    def gradient(self, x, t):
        
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        # print(dout.shape)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            # print('dout.shape : '+str(dout.shape))
            dout = layer.backward(dout)
        # print(' ')
        # 設定
        # ここの書き方
        # 重みとバイアスを更新すべき微小量の計算
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        # print('grads["W1"].shape : '+str(grads['W1'].shape))
        # print('grads["W1"] : '+str(grads["W1"]))
        grads['b1'] = self.layers['Affine1'].db
        # print('grads["b1"].shape : '+str(grads['b1'].shape))
        # print('grads["b1"] : '+str(grads["b1"]))
        grads['W2'] = self.layers['Affine2'].dW
        # print('grads["W2"].shape : '+str(grads['W2'].shape))
        # print('grads["W2"] : '+str(grads["W2"]))
        grads['b2'] = self.layers['Affine2'].db
        # print('grads["b2"].shape : '+str(grads['b2'].shape))
        # print('grads["b2"] : '+str(grads["b2"]))
             
        return grads
    









