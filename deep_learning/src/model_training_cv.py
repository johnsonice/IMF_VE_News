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
from data_sampling_utils import split_by_country, create_map,save_label_map
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
                        
#%%
if __name__ == '__main__':

    crisis_version = 'kr' #or 'rr'
    label_map_path = os.path.join(config.TRAINED_DEEP_MODEL,
                                  'cv_nn_model',
                                  'label_map.json')
    ## read training data 
    training_data_path = os.path.join(config.CRISIS_DATES,'train_data_{}.pkl'.format(crisis_version))
    df = pd.read_pickle(training_data_path)
    #%%
    dummies = df[['crisis_pre','crisis_tranqull','crisisdate']]
    df['label'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
    ## create label to description map 
    label2id, id2label = create_map(df,'label')
    df['label_num'] = df['label'].map(label2id)
    #%%
    ## get train test data iter
    n_fold_data = split_by_country(df,'snip_emb','label_num',n_fold=4)
    ## train data with cross validation
    for i in n_fold_data.keys():
        #i = 0 
        train_data_iter = torch.utils.data.DataLoader(n_fold_data[i]['train'], batch_size=16,shuffle=True)
        test_data_iter = torch.utils.data.DataLoader(n_fold_data[i]['test'], batch_size=16,shuffle=True)
        
        ## define model prams 
        input_size = 768
        hidden_size = [256,32]  ## curreently saved model weights  dropout_p = 0.1
        #hidden_size = [384,128,32]
        num_classes = 3
        learning_rate = 2e-4
        n_epochs = 10
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Simple_nn_model(input_size,hidden_size,num_classes,dropout_p=0.3,batchnorm=True,layernorm=True)
        #model = Dynamic_simple_sequencial_model(input_size,hidden_size,num_classes,dropout_p=0.4).model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
        
        ## train model 
        model,_,return_metric_df = train_model(model,optimizer,criterion,n_epochs,train_data_iter,
                                do_eval=True, test_data_iter= test_data_iter,
                                save_criterion='test_acc',warmup_n_epoch=0,  #'train_acc',None
                                save_model_path=os.path.join(config.TRAINED_DEEP_MODEL,
                                                             'cv_nn_model',
                                                             'weights_cv_{}'.format(i))) ## save model with cv id
        return_metric_df[['train_loss','test_loss']].plot(title='Training Metric')
    
    ## save the label map in the end 
    save_label_map(id2label,label_map_path) 
#    #%%
    
#    #%%
#    _,_,acc = eval_model(model,test_data_iter)
#    print(acc)
