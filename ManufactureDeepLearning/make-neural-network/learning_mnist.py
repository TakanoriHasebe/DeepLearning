#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 10:14:57 2017

@author: Takanori
"""

"""
MNISTを2層のニューラルネットワークで学習する
"""

import sys, os
sys.path.append('../common')  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('../dataset')  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from mnist import load_mnist
from multi_layer_network import MultiLayerNetwork # 勾配を算出する関数
from make_trainer import Trainer # ニューラルネットの訓練を行う関数

# datasetの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# networkの初期化
network = MultiLayerNetwork(784, 150, 10)

# 訓練クラスの呼び出し
trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=20, verbose=False)

# 実際の訓練
trainer.train()




