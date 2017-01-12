#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:21:35 2017

@author: Takanori
"""

"""
手書き画像認識

バッチ未使用
"""
import math
import numpy as np
from mnist import load_mnist
from PIL import Image #画像の表示に必要となる
from zodbpickle import pickle
   
def img_show(img):
    
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
        
#最初の呼び出しは少し待つ
#(訓練画像, 訓練ラベル), (テスト画像, テストラベル)
def get_data():
    
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(flatten=True, normalize=False, one_hot_label=False) #flatten : 入力画像を平らにする, normalize : 入力画像の正規化(0.0 ~ 1.0)しない, one_hot_label : [0, 0, ..., 1]な
    
    return x_test, t_test
"""
#それぞれのデータの形式を出力
print(x_train.shape)
print(t_train.shape)
print(x_test.shape) 
print(t_test.shape)    
""" 
"""
img = x_train[0]
label = t_train[0]
print('label : '+str(label))

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
"""
#img_show(img) #画像を表示する

#sigmoid関数でoverflowが発生
def sigmoid(x):
    
    return 1 / (1 + np.exp(-x))
    
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #オーバーフローを回避
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y    

#学習済みのニューラルネット
def init_network():
    
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        
    return network

#学習済みのニューラルネットの出力    
def predict(network, x):
    
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
 
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    
    y = predict(network, x[i])
    p = np.argmax(y) #最も確率の高いものを取得
    if p == t[i]:
        accuracy_cnt += 1
    
print("Accuracy:"+str(float(accuracy_cnt) / len(x)))


    
    
    
    
    
    
    
    
    
    