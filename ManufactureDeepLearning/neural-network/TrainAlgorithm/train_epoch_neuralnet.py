#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:50:20 2017

@author: Takanori
"""

"""
ミニバッチ学習を実装する。
"""

import sys
sys.path.append('../../')

'''
from common.gradientfunctions import numerical_gradient
from common.outputactivationfunctions import softmax
from common.lossfunctions import cross_entropy_error
from common.activatingfunctions import sigmoid
'''
import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from zodbpickle import pickle

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

## 付け加え
train_acc_list = []
test_acc_list = []
# 1epochあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print(i)
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    # grad = network.gradient(x_batch, t_batch) # 高速版
    
    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
# 1epochごとに認識精度を計算
if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc | "+str(train_acc)+", "+str(test_acc))

print('len(train_loss_list):'+str(len(train_loss_list)))
pickle.dump(train_loss_list, open('train_loss_list.pkl','wb'), protocol=3 )