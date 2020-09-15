#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:15:44 2020

@author: chuang
"""
## simple model evaluation 
import os,json
import config
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from infer_utils import Inference_model,get_precision,get_recall,get_fscore
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

    #%%
    
if __name__ == '__main__':
    crisis_version = 'kr'
    trained_model_path=os.path.join(config.TRAINED_DEEP_MODEL,
                                    'simple_nn_model',
                                    'weights_simple')
    
    label_map_path = os.path.join(config.TRAINED_DEEP_MODEL,
                                  'simple_nn_model',
                                  'label_map.json')
    
    training_data_path = os.path.join(config.CRISIS_DATES,
                                      'train_data_{}.pkl'.format(crisis_version))
   #%% 
    M = Inference_model(trained_model_path,label_map_path,load_mode='full')
#%%
    ## read training data 

    df = pd.read_pickle(training_data_path)
    dummies = df[['crisis_pre','crisis_tranqull','crisisdate']]
    df['label'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])

    X = df['snip_emb']
    res_labels,res_names = M.predict_from_emb(X)
    
#%%
    df['predict_label'] = res_labels
    df['predict_name'] = res_names

#%%
    ## recode crisisdates to crisis pre_dates too
    df.predict_name = df.predict_name.replace('crisisdate','crisis_pre') ## here we do not ditinguish crisis and precrisis 
    #%%
    df['correct']  = df['predict_name'] == df['label']
    tp = df['correct'][df.crisis_pre == 1].sum()
    fp = len(df['correct'][df.crisis_tranqull == 1]) - df['correct'][df.crisis_tranqull == 1].sum()
    df['value_group'] = (df.groupby('country_name').crisis_pre.diff(1)!=0).astype('int').cumsum() ## by country, code each precrisis period
    pre_crisis_signal_counts = df[df.crisis_pre==1].groupby('value_group').correct.sum()
    fn = len(pre_crisis_signal_counts) - pre_crisis_signal_counts.astype(bool).sum()
    
    precision = get_precision(tp,fp)
    recall = get_recall(tp,fn)
    f2 = get_fscore(tp,fp,fn,beta=2)
    
    print(precision,recall,f2)