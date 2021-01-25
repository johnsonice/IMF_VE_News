#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 05:23:20 2020

@author: chuang

filter news data from raw and merge crisis dates with news data

"""

import pandas as pd 
import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os
from utils import map_df_value
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#%%

def filterby_news_signal(train_df,signal_col='crisisdate_news'):
    
    #train_df['crisisdate'] = train_df[signal_col]*train_df['crisisdate']
    train_df['crisis_pre'] = train_df['crisis_pre']*train_df[signal_col]
    train_df['crisis_tranqull'] = 1
    train_df['crisis_tranqull'] = train_df['crisis_tranqull'] - train_df['crisisdate'] - train_df['crisis_pre']
    summary = train_df.describe()
    logger.info('after merging with News index signal')
    logger.info(summary.loc['mean'])
    
    return train_df
    
def process_kr(news_crisis_path,training_data_path_kr,keep_signal_only=True):
    """
    combine kr data with news index 
    """
    df_crisis_dates_news = pd.read_pickle(news_crisis_path)
    df_crisis_dates_kr = pd.read_pickle(training_data_path_kr)
    train_df= df_crisis_dates_kr.merge(df_crisis_dates_news[['ifscode','month','crisisdate_news']],
                                 how='inner', 
                                 left_on=['month','ifs_code'],
                                 right_on=['month','ifscode'])
    summary = train_df[['crisisdate','crisis_pre','crisis_tranqull','crisisdate_news']].describe()
    logger.info('Before merging with News index signal')
    logger.info(summary.loc['mean'])
    
    if keep_signal_only:
        train_df = filterby_news_signal(train_df,signal_col='crisisdate_news')
    
    return train_df


#%%
if __name__ == '__main__':
    
    ## set up global arguments
    read_data=True
    window_size = 12
    #run_rr = False
    run_kr = True
    #run_ve_q = False
    
    news_crisis_path = os.path.join(config.CRISIS_DATES,'criris_dates_news_index.pkl')
    
    training_data_path_kr = os.path.join(config.CRISIS_DATES,'train_data_kr_w{}.pkl'.format(window_size))
    #training_data_path_rr = os.path.join(config.CRISIS_DATES,'train_data_rr.pkl')
    #training_data_path_ve_q = os.path.join(config.CRISIS_DATES,'train_data_ve_q.pkl')
    
    kr_news = os.path.join(config.CRISIS_DATES,'train_data_kr_news_filter_w{}.pkl'.format(window_size))
    #rr_news = os.path.join(config.CRISIS_DATES,'train_data_rr_news_filter.pkl')
    #ve_q_news = os.path.join(config.CRISIS_DATES,'train_data_ve_q_news_filter.pkl')
    
    #%%
    if run_kr:
        train_df = process_kr(news_crisis_path,training_data_path_kr)
        train_df.to_pickle(kr_news)
        logger.info('export data to {}'.format(kr_news))


