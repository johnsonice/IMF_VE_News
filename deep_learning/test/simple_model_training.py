#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:04:21 2020

@author: chuang
"""

## train simple model 
import torch 
import torch.nn as nn
import pandas as pd
import config
import os
import numpy as np
from sklearn.model_selection import train_test_split

def create_map(df,label_column):
    label2id = {}
    for idx,key in enumerate(df[label_column].unique()):
        label2id[key] = idx
        
    id2label = {i:key for key,i in label2id.items()}
    
    return label2id, id2label

def prepare_torch_training_data(df,x_label,y_label,ratio=0.3):
    """
    prepare data into torch dataset for training
    """
    
    X = df[x_label]
    y = df[y_label]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =ratio,random_state=0)

    train_data = list(zip(X_train,y_train))
    test_data = list(zip(X_test,y_test))

    train_data_iter = torch.utils.data.DataLoader(train_data, batch_size=16,shuffle=True)
    test_data_iter = torch.utils.data.DataLoader(test_data, batch_size=16,shuffle=True)
    #next(iter(train_data_iter))
    
    return train_data_iter,test_data_iter
    
#%%
if __name__ == '__main__':

    ## read training data 
    training_data_path = os.path.join(config.CRISIS_DATES,'train_data.pkl'.format())
    df = pd.read_pickle(training_data_path)
    dummies = df[['crisis_pre','crisis_tranqull','crisisdate']]
    df['label'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
    ## create label to description map 
    label2id, id2label = create_map(df,'label')
    df['label_num'] = df['label'].map(label2id)
    ## get train test data iter
    train_data_iter,test_data_iter = prepare_torch_training_data(df,'snip_emb','label_num',ratio=0.3)
    
    
    
    