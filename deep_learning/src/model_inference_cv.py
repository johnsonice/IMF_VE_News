#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 05:46:34 2020

@author: chuang
"""

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
from utils import write_to_txt
import logging
import pickle
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
    
        #%%
    
if __name__ == '__main__':
    crisis_version = 'kr'
    training_data_path = os.path.join(config.CRISIS_DATES,
                                      'train_data_{}.pkl'.format(crisis_version))
    cv_data_path = os.path.join(config.CRISIS_DATES,
                                      'train_data_cv_{}.pkl'.format(crisis_version))
    ## load cv data
    with open(cv_data_path,'rb') as f:
       cv_data = pickle.load(f)
    

    trained_model_paths={i:os.path.join(config.TRAINED_DEEP_MODEL,
                                    'cv_nn_model',
                                    'weights_cv_{}'.format(i)) for i in cv_data.keys()}
    
    label_map_path = os.path.join(config.TRAINED_DEEP_MODEL,
                                  'cv_nn_model',
                                  'label_map.json')
    
    model_eval_result = os.path.join(config.TRAINED_DEEP_MODEL,
                                  'cv_nn_model',
                                  'eval.txt')
    
   #%% 
    tps = 0
    fps = 0
    fns = 0
    write_to_txt(model_eval_result,'CV model Results',over_write=True)
    for i in cv_data.keys():
        #i = 1
        M = Inference_model(trained_model_paths[i],label_map_path,load_mode='full')
    #%%
        ## read training data
        df = pd.read_pickle(training_data_path)
        dummies = df[['crisis_pre','crisis_tranqull','crisisdate']]
        df['label'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
        df = df[-df.country_name.isin(cv_data[i]['countries'])]
        
        #%%
        X = df['snip_emb'].tolist()
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
        
        tps += tp
        fps += fp
        fns += fn
        
        precision = get_precision(tp,fp)
        recall = get_recall(tp,fn)
        f2 = get_fscore(tp,fp,fn,beta=2)
        msg = '{}\nprecision:{} ; recall:{} ; f2:{}'.format(cv_data[i]['countries'],precision,recall,f2)
#        print(cv_data[i]['countries'])
#        print('precision:{} ; recall:{} ; f2:{}'.format(precision,recall,f2))
        write_to_txt(model_eval_result,msg,over_write=False)
        print(msg)
    precision = get_precision(tps,fps)
    recall = get_recall(tps,fns)
    f2 = get_fscore(tps,fps,fns,beta=2)
    msg = 'Overall results\nprecision:{} ; recall:{} ; f2:{}'.format(precision,recall,f2)
    write_to_txt(model_eval_result,msg,over_write=False)
#    print('Overall Results')
#    print('precision:{} ; recall:{} ; f2:{}'.format(precision,recall,f2))
    print(msg)