#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 17:05:53 2017

@author: Takanori
"""

"""
MNISTの学習

mini_batchを用いている

ミニバッチ学習は, 全体のおおよその近似として, ランダムに選ばれた小さな集まり（ミニバッチ）で代替するということ
"""
import math
import numpy as np
from mnist import load_mnist
from PIL import Image #画像の表示に必要となる
from zodbpickle import pickle

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

#print(x_train.shape) # 訓練データは60000個あり, 784次元
#print(t_train.shape) # 教師データは60000個あり, 10次元

# ミニバッチ学習を行う為に, 訓練データからランダムに10枚抜き出す
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 60000個の中から10個ランダムに選ぶ
x_batch = x_train[batch_mask] # 学習データの作成
t_batch = t_train[batch_mask] # 教師データの作成
#print(x_batch.shape) # 訓練データ10個, 784次元
#print(t_batch.shape) # 教師データ10個, 10次元





