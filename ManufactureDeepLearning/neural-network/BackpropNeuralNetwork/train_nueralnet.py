#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:09:17 2017

@author: Takanori
"""

"""
誤差逆伝搬法を用いいたMNISTの学習
"""

import sys
sys.path.append('../MNIST/')
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from zodbpickle import pickle


# データの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)   
    
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

# 勾配最適化で用いるパラメータ
# 確率的勾配降下法
learning_rate = 0.1
# AdaGrad
lr = 0.01
h = None


train_loss_list = list()
train_acc_list = list()
test_acc_list = list()

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 誤差逆伝搬法によって勾配を算出
    grad = network.gradient(x_batch, t_batch)
    
    """
    # 更新
    # 確率的勾配降下法
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    """  
    
    # 更新
    # AdaGrad
    for key in ('W1', 'b1', 'W2', 'b2'):
        
        # 変数の初期化
        if h is None:
            h = {}
            for key, val in network.params.items():
                h[key] = np.zeros_like(val)
                
        # 更新
        for key in network.params.keys():
            h[key] = grad[key] * grad[key]
            network.params[key] -= lr*grad[key] / (np.sqrt(h[key]) + 1e-7)
        
        
    loss = network.loss(x_batch, t_batch)
    
    # print(x_batch, t_batch)
    # print('x_batch : '+str(x_batch)+'t_batch : '+str(t_batch)+'loss : '+str(loss))
    # print(loss)
    train_loss_list.append(loss)
    # print(len(train_loss_list))
    
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
    
# pickle.dump(train_acc, open('train_accuracy.pkl','wb'), protocol=3 )
# pickle.dump(test_acc, open('test_accuracy.pkl','wb'), protocol=3 )




