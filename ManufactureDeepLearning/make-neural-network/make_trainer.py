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
6. ミニバッチ学習の際の繰り返しの回数の設定について
7. ミニバッチ学習とバッチ学習
8. train_step関数とtrain関数
"""

import sys, os
sys.path.append('../')  # 親ディレクトリのファイルをインポートするための設定
from common import optimizer
import numpy as np
from multi_layer_network import MultiLayerNetwork

# ニューラルネットワークの訓練を行うクラス
class Trainer:
    
    def __init__(self, network, x_train, t_train, x_test, t_test, mini_batch_size=100, epochs=20, verbose=True):
        
        self.network = network # MultiLayerNet
        self.x_train = x_train # 学習データ
        self.t_train = t_train # 学習データ
        self.x_test = x_test # テストデータ
        self.t_test = t_test # テストデータ
        self.batch_size = mini_batch_size # バッチ学習のサイズ
        self.train_size = x_train.shape(0) # 学習する配列のサイズ
        self.optimizer = optimizer.SGD # 最適化にSGDを用いる
        self.verbose = verbose # 冗長性
        self.train_size = train_size.shape[0] # 学習サイズ 
        self.current_iter = 0 # 現在の繰り返し回数
        self.iter_per_epoch = max(train_size/mini_batch_size, 1) # 0になることを避ける。ミニバッチ学習の繰り返し回数
        self.current_epoch = 0 # 現在のepoch数
        self.max_iter = epochs * self.iter_per_epoch
        
    
        self.train_loss_list = list() # 学習時の誤差リスト
        self.train_acc_list = list() # 学習データの正確さ
        self.test_acc_list = list() # テストデータの正確さ
        
        
    # 学習の流れ                                  
    def train_step(self):
        # バッチ学習のデータの用意
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        # 勾配の算出
        grads = self.network.gradient(x_batch, t_batch)
        
        # 勾配の更新
        self.optimizer.update(self.network.params, grads)       

        # 誤差の算出
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        
        # 冗長性がTrueであれば表示
        if self.verbose : print('train loss:'+str(loss))
        
        # ミニバッチ学習
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            self.train_acc_list.append(self.network.accuracy(self.x_train, self.t_train))
            self.test_acc_list.append(self.network.accuracy(self.x_test, self.t_test))
            
            if self.verbose : print("=== epoch:" + str(self.current_epoch)+', train acc:'+str(self.train_acc_list[self.current_epoch - 1])+', test acc'+str(self.test_acc_list[self.current_epoch - 1]))
        
        self.current_iter += 1
            
    # 訓練
    def train(self):

        for i in range(self.max_iter):
            self.train_step()
        
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))
