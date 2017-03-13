#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 14:41:46 2017

@author: Takanori
"""

"""
chainerを用いてmnistデータを学習する
"""
import time
import sys, os
sys.path.append('../common')  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('../dataset')  # 親ディレクトリのファイルをインポートするための設定
from mnist import load_mnist
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, \
            Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from zodbpickle import pickle

# set data
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=False)
    
x_train = Variable(np.array(x_train, dtype=np.float32))
t_train = Variable(np.array(t_train, dtype=np.int32))
x_test = Variable(np.array(x_test, dtype=np.float32))
t_test = Variable(np.array(t_test, dtype=np.int32))

# print('t_test.data.shape:'+str(t_test.data.shape))
# print('t_test.data:'+str(t_test.data))

# print(x_train.shape)
# print(t_train.shape)

start = time.time()

# Define model
class MyChain(Chain):
    
    def __init__(self):
        super(MyChain, self).__init__(
            l1 = L.Linear(784, 150),
            l2 = L.Linear(150, 10)
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.fwd(x), t)
    
    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

# Initializa model
model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

# Learn
mini_batch_size = 100
epochs = 20
train_size = x_train.shape[0]
iter_per_epoch = int(max(x_train.shape[0]/mini_batch_size, 1)) # 0になることを避ける。ミニバッチ学習の繰り返し回数
max_iter = epochs*iter_per_epoch

acc_list = list()
for i in range(50):
    
    print('epoch:'+str(i))
    for j in range(max_iter):
        # print(j)
        batch_mask = np.random.choice(train_size, mini_batch_size)
        x_batch = x_train.data[batch_mask]
        t_batch = t_train.data[batch_mask]
        
        x = Variable(np.array(x_batch, dtype=np.float32))
        t = Variable(np.array(t_batch, dtype=np.int32))
    
        # print(x.data.shape)
        # print(t.data.shape)
        
        model.zerograds()
        
        loss = model(x, t)
        loss.backward()
        
        optimizer.update()
        
    ## テストデータに対する効率を算出する ## 
    y = model.fwd(x_test)
    # print('x_test.shape:'+str(x_test.shape))
    # print('y.shape:'+str(y.shape))
    # print('y:'+str(y.data))
    res = np.argmax(y.data, axis=1)
    # print('res:'+str(res))
    accuracy = (np.sum(res == t_test.data) / res.shape[0]) * 100
    print('正解率:'+str(accuracy))
    acc_list.append(accuracy)

elapsed_time = time.time() - start                 
# print(elapsed_time)
# print(accuracy_list)

pickle.dump(acc_list, open('acc_list_chainer.pkl','wb'), protocol=3 )
pickle.dump(elapsed_time, open('elapsed_time_chainer.pkl','wb'), protocol=3 )



"""
n = train_size   
bs = 25   
for j in range(5000):   
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(x_train[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = Variable(t_train[sffindx[i:(i+bs) if (i+bs) < n else n]])
        model.zerograds()
        loss = model(x,y)
        loss.backward()
"""





