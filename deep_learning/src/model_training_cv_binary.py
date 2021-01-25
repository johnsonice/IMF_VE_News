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
import pickle
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
                        
#%%

def transform_df_with_labeles(df,
                              columns=['crisis_pre','crisis_tranqull','crisisdate'],
                              merge_class=False):
    """
    prepare label for training; check if we need to merge crisisdate with precrisis_date
    """
    ## create label column
    dummies = df[columns]
    df['label'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
    
    ## merge categories 
    if merge_class:
        df['label'] = df['label'].replace('crisisdate','crisis_pre')
    else:
        pass
    
    ## create label to description map 
    label2id, id2label = create_map(df,'label')
    df['label_num'] = df['label'].map(label2id)
    
    return df,label2id,id2label
    
 
#%%
if __name__ == '__main__':
    
    ##############################
    ## specify global arguments ##
    ##############################
    crisis_version = 'kr_news_filter_w12' #'kr_news_filter_w12'#'kr_w12' #or 'rr' or 've_q' or 'kr' or 'kr_w3-12'
    n_class = 2 
    if n_class == 2:
        merge_class = True
    else:
        merge_class = False
    
    label_map_path = os.path.join(config.TRAINED_DEEP_MODEL,
                                  'cv_nn_model',
                                  'label_map_{}_c{}.json'.format(crisis_version,n_class))
    
    ## specify data path  to export 
    cv_data_path = os.path.join(config.CRISIS_DATES,'train_data_cv_{}_c{}.pkl'.format(crisis_version,n_class))
    ## read training data 
    training_data_path = os.path.join(config.CRISIS_DATES,'train_data_{}.pkl'.format(crisis_version))
    df = pd.read_pickle(training_data_path)
    
    ##################################
    ## prepare training data format ##
    ##################################
    #%%
    df,label2id,id2label = transform_df_with_labeles(df,
                              columns=['crisis_pre','crisis_tranqull','crisisdate'],
                              merge_class=merge_class)
    
    #%%
    ## get train test data iter
    n_fold_data = split_by_country(df,'snip_emb','label_num',panel_id='country_name',n_fold=4)
    #%%
    ##################################
    ## Train with cross validation ###
    ##################################
    ## train data with cross validation
    for i in n_fold_data.keys():
        #i = 0 
        print('n folder: {}'.format(i))
        train_data_iter = torch.utils.data.DataLoader(n_fold_data[i]['train'], batch_size=16,shuffle=True,drop_last=True)
        test_data_iter = torch.utils.data.DataLoader(n_fold_data[i]['test'], batch_size=16,shuffle=True)
        
        ## define model prams 
        input_size = 768
        hidden_size = [256,32]  ## curreently saved model weights  dropout_p = 0.1
        #hidden_size = [384,128,32]
        num_classes = n_class
        learning_rate = 2e-4
        n_epochs =15
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Simple_nn_model(input_size,hidden_size,num_classes,dropout_p=0.3,batchnorm=True,layernorm=True)
        #model = Dynamic_simple_sequencial_model(input_size,hidden_size,num_classes,dropout_p=0.4).model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
        
        ## train model 
        model,_,return_metric_df = train_model(model,optimizer,criterion,n_epochs,train_data_iter,
                                do_eval=True, test_data_iter= test_data_iter,
                                save_criterion=None,warmup_n_epoch=0,  #'train_acc',None
                                save_model_path=os.path.join(config.TRAINED_DEEP_MODEL,
                                                             'cv_nn_model',
                                                             'weights_cv_{}_c{}_{}'.format(crisis_version,
                                                                                             n_class,
                                                                                             i))) ## save model with cv id
        
        return_metric_df[['train_loss','test_loss']].plot(title='Training Metric')
    #%%
    ## save the label map in the end 
    save_label_map(id2label,label_map_path) 
    with open(cv_data_path,'wb') as f:
        pickle.dump(n_fold_data,f)
    logger.info('eexport to {}'.format(cv_data_path))
#    #%%
    
#    #%%
#    _,_,acc = eval_model(model,test_data_iter)
#    print(acc)
