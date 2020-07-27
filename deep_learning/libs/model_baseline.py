#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:31:37 2020

@author: chuang
"""

####baseline models 

import torch 
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
#import os
import numpy as np
from collections import OrderedDict
#from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Simple_nn_model(nn.Module):
    """
    a fix feed forward NN with 2 hidden layers 
    hiden_zise is a list of two element; emg [128,64]
    """
    def __init__(self,input_size,hidden_size,num_classes,dropout_p=0.1,batchnorm=True,layernorm=False):
        super(Simple_nn_model,self).__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1=nn.Linear(input_size,hidden_size[0])
        self.fc2=nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3=nn.Linear(hidden_size[1],num_classes)
        if batchnorm:
            self.bn0 = nn.BatchNorm1d(num_features = input_size)
            self.bn1 = nn.BatchNorm1d(num_features = hidden_size[0])
            self.bn2 = nn.BatchNorm1d(num_features = hidden_size[1])
        if layernorm:
            self.ln0 = nn.LayerNorm(input_size)
            self.ln1 = nn.LayerNorm(hidden_size[0])
            self.ln2 = nn.LayerNorm(hidden_size[1])
            
    def forward(self,x):
        
        ## before fc1
        if self.batchnorm:
            x = self.bn0(x)
        if self.layernorm:   
            x = self.ln0(x)
            
        ## fc1
        out = self.fc1(self.dropout(x))
        if self.batchnorm:
            out=self.bn1(out)
        if self.layernorm:
            out=self.ln1(out)
            
        out = self.dropout(F.relu(out))
        
        ## fc2
        out = self.fc2(out)
        if self.batchnorm:    
            out =self.bn2(out)
        if self.layernorm:
            out =self.ln2(out)
    
        out=F.relu(out)

        ## fc3
        out = self.fc3(out)
        
        return out 

class Dynamic_simple_sequencial_model(object):
    """
    hidden_size can be a dynamic list with any numb of elements e.g [128,64,32,4]
    we can also use modellist instead of sequencial. it should be the same
        pytorch.org/docs/master/generated/torch.nn.ModuleList.html
    """
    def __init__(self,input_size,hidden_size,num_classes,dropout_p=0.1,batchnorm=True,layernorm=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.batchnorm=batchnorm
        self.layernorm=layernorm
        self.model_structure = self.construct_model_structure_dict()
        self.model = nn.Sequential(self.model_structure)
        
    
    def construct_model_structure_dict(self):
        in_out_size_list = [self.input_size] + self.hidden_size+[self.num_classes]
        in_out_size_pires = [in_out_size_list[i:i+2] for i in range(len(in_out_size_list)-1)]
        structure = []
        for idx,p in enumerate(in_out_size_pires):
            if idx == len(in_out_size_pires)-1:
                last_layer=-1  ## only fc layer
            elif idx == len(in_out_size_pires)-2:
                last_layer = -2 ## fc + norm + activateion 
            else:
                last_layer = 0 ## fc + norm + activateion + dropout 
                
            structure += self._construct_layer(p,idx,last_layer)
        
        structure = OrderedDict(structure)
        return structure
    
    
    def _construct_layer(self,in_out_size_pair,sequence_id,last_layer=0):
        layer = [('fc{}'.format(sequence_id),nn.Linear(in_out_size_pair[0],in_out_size_pair[1]))]
        if last_layer == -1:
            pass
        elif last_layer == -2:
            layer += [
                        ('bn{}'.format(sequence_id),nn.BatchNorm1d(num_features=in_out_size_pair[1])),
                        ('act{}'.format(sequence_id),nn.ReLU()),
                    ]
        else:
            layer += [
                        ('bn{}'.format(sequence_id),nn.BatchNorm1d(num_features=in_out_size_pair[1])),
                        ('act{}'.format(sequence_id),nn.ReLU()),
                        ('dropout{}'.format(sequence_id),nn.Dropout(p=self.dropout_p)),
                    ]
        
        return layer 
            
#%%
if __name__ == '__main__':
    input_size = 768
    hidden_size = [123,12,24]
    num_classes = 3
    
    dynamic_m = Dynamic_simple_sequencial_model(input_size,hidden_size,num_classes) 
    print(dynamic_m.model)
    
    
    