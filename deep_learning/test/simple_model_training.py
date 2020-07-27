#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:04:21 2020

@author: chuang
"""

## train simple model 
import os,sys
sys.path.insert(0,'../libs')
from model_baseline import Simple_nn_model,Dynamic_simple_sequencial_model
from train_utils import train_model
import config
import torch 
import torch.nn as nn
#from torch.nn import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    train_data_iter,test_data_iter = prepare_torch_training_data(df,'snip_emb','label_num',ratio=0.4)
    
    #%%
    
    input_size = 768
    hidden_size = [256,32]  ## curreently saved model weights  dropout_p = 0.1
    #hidden_size = [384,128,32]
    num_classes = 3
    learning_rate = 1e-3
    n_epochs = 30
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Simple_nn_model(input_size,hidden_size,num_classes,dropout_p=0.3,batchnorm=True,layernorm=True)
    #model = Dynamic_simple_sequencial_model(input_size,hidden_size,num_classes,dropout_p=0.4).model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
    
    #%%
    
    model,_,return_metric_df = train_model(model,optimizer,criterion,n_epochs,train_data_iter,
                            do_eval=True, test_data_iter= test_data_iter,
                            save_criterion=None,warmup_n_epoch=0,  #'train_acc'
                            save_model_path=os.path.join(config.TRAINED_DEEP_MODEL,
                                                         'simple_nn_model',
                                                         'weights_simple.pt'))
#    
    return_metric_df[['train_loss','test_loss']].plot(title='Training Metric')
#    #%%
#    _,_,acc = eval_model(model,test_data_iter)
#    print(acc)
