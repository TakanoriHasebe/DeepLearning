#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:57:36 2017

@author: Takanori
"""

"""
ニューラルネットワークの訓練を行うクラス
* 課題点
0. 初期化関数群
1. バッチ処理について忘れている
2. バッチ処理とミニバッチ処理について
3. バッチ処理の書き方
4. 勾配の最適化手法の初期化
5. 勾配の更新でパラメータをどこから持ってくるかについて

"""

import sys, os
sys.path.append('../common')  # 親ディレクトリのファイルをインポートするための設定
from common import optimizer
import numpy as np
from multi_layer_network import MultiLayerNetwork

# ニューラルネットワークの訓練を行うクラス
class Trainer:
    
    def __init__(self, network, x_train, t_train, mini_batch_size=100):
        
        self.network = network # MultiLayerNet
        self.x_train = x_train # train_step関数で用いる
        self.t_train = t_train # train_step関数で用いる
        self.batch_size = mini_batch_size # バッチ学習のサイズ
        self.train_size = x_train.shape(0) # 学習する配列のサイズ
        self.optimizer = optimizer.SGD
        
                                       
    def train_step(self):
        # バッチ学習のデータの用意
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        # 勾配の算出
        grads = self.network.gradient(x_batch, t_batch)
        
        # 勾配の更新
        self.optimizer.update(self.network.params, grads)       








