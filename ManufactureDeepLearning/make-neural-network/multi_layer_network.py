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
５層のニューラルネットワークを設計する
"""

import sys, os
sys.path.append('../common')  # 親ディレクトリのファイルをインポートするための設定
from layers import Sigmoid, Affine, SoftmaxWithLoss # 誤差逆伝搬の時に必要
from collections import OrderedDict

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
    def __init__(self, input_size, output_size):
        
        pass
        
    
        
        







