#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:25:35 2017

@author: Takanori
"""

"""
最初はMNISTに対して作成される。
誤差逆伝搬法を用いて学習。
多層のニューラルネットワークである。
５層のニューラルネットワークを自分で設計する

# 課題点
0. 重み, バイアスのパラメータと層の設計はクラス内の他の関数でも使うので, selfにすべき
1. クラス内の初期化関数に重み, バイアス, Layerの追加を行っていなかった。
2. バイアスの配列の形, また最初random.randnを用いたが, zerosで初期化すべきだった
3. numpyにおいての内積とバイアスの足し算
4. 初期化関数内での活性化関数の初期化の仕方
5. 出力層のsoftmaxと誤差関数をself.lastLayer = ..()で初期化すべき
6. 順伝搬時のpredict関数でのfor文の動き
7. 辞書内にクラスのインスタンスを作成した時, 辞書のクラス内の関数の呼び出し方
8. MultiLayerNetworkクラスでは勾配を求める
9. ニューラルネットワークの全体の設計図がわかっていない
10. 逆伝搬時の層を反対にするところ
11. 最終的に勾配を返したいので, 勾配の辞書を作成する
参考url : https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/multi_layer_net.py
"""

import sys, os
sys.path.append('../common')  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('../dataset')  # 親ディレクトリのファイルをインポートするための設定
from layers import Sigmoid, Affine, SoftmaxWithLoss # 誤差逆伝搬の時に必要
from collections import OrderedDict
import numpy as np
from mnist import load_mnist


# 多層ニューラルネットワークの設計
# 最初は２層で設計してみる
# 用いる層はAffine_layerとSigmoidで重みの初期化に気をつける
class MultiLayerNetwork:
    
    # 層の設定
    """
    input_size : 学習データの入力サイズ
    hidden_size : 隠れ層のサイズ
    output_size : NNの出力サイズ
    """
    def __init__(self, input_size, output_size, hidden_size):
        
        # 重み, バイアス
        self.params = {}
        # 入力層の重み, バイアスの設定
        self.params['W1'] = np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        # 出力層の重み, バイアスの設定
        self.params['W2'] = np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # ニューラルネットワークの層を設計する
        self.layers = OrderedDict()
        # 入力層のAffineレイヤ
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        # Sigmoidレイヤ
        self.layers['Sigmoid1'] = Sigmoid() 
        # 出力層のAffineレイヤ
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        # 出力層と誤差関数(corss_entropy_error)
        self.lastLayer = SoftmaxWithLoss()
    
    # lastLayerまでの計算
    def predict(self, x):
        
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x
     
    # 出力から誤差を算出
    def loss(self, x, t):
        
        # predictを書くことで, lossのみの呼び出し
        y = self.predict(x)
        
        # 誤差の算出
        out = self.lastLayer.forward(y, t)
        
        return out
    
    # 正確さを算出
    # ミニバッチ対応
    # このクラス内では用いない
    def accuracy(self):
        
        pass
    
    # 勾配を算出する関数
    def gradient(self, x, t):
        
        # 順伝搬
        self.loss(x, t)
        
        #逆伝搬
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        # ここの書き方
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layers.backward(dout)
            
        # 勾配を返す
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
            
        return grads
        
# データセットの読み込み    
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
        








