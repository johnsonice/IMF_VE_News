#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:37:38 2020

@author: chuang
"""

## data sampling utils 
#import torch 
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
import json

def chunks(lst,n,rd=False):
    """Yield successive n-sizied chunks from 1st"""
    div_n = len(lst)/n
    if rd: ## random shuffle or not 
        lst = random.shuffle(lst)
        
    res = [lst[int(round(div_n*i)):int(round(div_n*(i+1)))] for i in range(n)]

    return res

def create_map(df,label_column):
    """create label map for string column"""
    label2id = {}
    for idx,key in enumerate(df[label_column].unique()):
        label2id[key] = idx
        
    id2label = {i:key for key,i in label2id.items()}
    
    return label2id, id2label

def save_label_map(label2id,map_path):
    """Save json file"""
    with open(map_path,'w') as fp:
        json.dump(label2id,fp)
    
    return None


def ramdom_split(df,x_label,y_label,ratio=0.3):
    """
    prepare data into torch dataset for training
    """
    
    X = df[x_label]
    y = df[y_label]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =ratio,random_state=0)

    train_data = list(zip(X_train,y_train))
    test_data = list(zip(X_test,y_test))
    
    return train_data,test_data

def split_by_country(df,x_label,y_label,panel_id='country_name',n_fold=4):
    """
    df : with country_name ; and all X and y data
    return: {
                0:{
                    'countries':[],
                    'train':[X,y],
                    'test':[X,y]
                },
                1:{
                    'countries':[],
                    'train':[X,y],
                    'test':[X,y]
                }
                ... ... ...
            }
    """
    res = {}
    countries = list(df[panel_id].unique())
    country_chunks = chunks(countries,n = n_fold)
    for idx,cs in enumerate(country_chunks):
        
        train = df[df[panel_id].isin(cs)]
        test = df[-df[panel_id].isin(cs)]
        train_X,train_y = train['snip_emb'].tolist(),train[y_label].tolist()
        test_X,test_y = test['snip_emb'].tolist(),test[y_label].tolist()
        res[idx]={}
        res[idx]['countries'] = cs
        res[idx]['train'] = list(zip(train_X,train_y))
        res[idx]['test'] = list(zip(test_X,test_y))
        
    return res
#%%
if __name__ == '__main__':
    ## read training data 
    CRISIS_DATES = '/data/News_data_raw/FT_WD/crisis_date'
    crisis_version = 'kr'
    
    training_data_path = os.path.join(CRISIS_DATES,'train_data_{}.pkl'.format(crisis_version))
    df = pd.read_pickle(training_data_path)

    res = split_by_country(df,'snip_emb','snip_emb',n_fold=4)
    
    print(res[0]['train'][0][0].shape)
    print(res[0]['train'][0][0].mean())
