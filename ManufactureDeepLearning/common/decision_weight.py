#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:23:20 2017

@author: Takanori
"""

"""
活性化関数に対して重みを決定する
"""
import numpy as np

class DecisionWeight:
    
    """
    def __init__(self, input_node_num, output_node_num):
        
        self.weight_linear = np.random.randn(input_node_num, output_node_num) / np.sqrt(input_node_num)
        self.weight_non_linear = np.random.randn(input_node_num, output_node_num) * np.sqrt(2.0 / input_node_num)
    """
    
    # 活性化関数の種類, 入力ノード, 出力ノード
    def decision(self, activation_type, input_node_num, output_node_num):
        
        weight_linear = np.random.randn(input_node_num, output_node_num) / np.sqrt(input_node_num)
        weight_non_linear = np.random.randn(input_node_num, output_node_num) * np.sqrt(2.0 / input_node_num)
        
        if activation_type == 'Sigmoid' or 'tanh':
            return weight_linear
        elif activation_type == 'ReLU':
            return weight_non_linear
