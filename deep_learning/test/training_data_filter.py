#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 05:23:20 2020

@author: chuang
"""

import pandas as pd 
import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#%%

def filter_country_feature(country_df,topn=10,emb_only=True):
    """
    filter top featured documents 
    """
    country_df.sort_values(by=['date','all_language'],ascending=(True,False),inplace=True)
    country_df = country_df.groupby('month').head(topn)
    if emb_only:
        country_df = country_df[['date', 'week', 'month', 
                                 'quarter','snip', 'title', 
                                 'snip_emb', 'title_emb']]
    country_df.dropna(inplace=True)
    return country_df 

def aggregate_emb(country_df):
    """
    average aggregate embeding in a month
    """
    new_df_data = []
    unique_months = country_df.month.unique()
    for i in unique_months:
        #logger.info(i)
        df_m=country_df[country_df['month']==i]
        title_emb = np.mean(np.array(df_m.title_emb.tolist()),axis=0)
        snip_emb = np.mean(np.array(df_m.snip_emb.tolist()),axis=0)
        new_df_data.append((i,title_emb,snip_emb))
    
    new_df = pd.DataFrame(new_df_data,columns=['month','title_emb','snip_emb'])
    return new_df

#%%
if __name__ == '__main__':
    
    read_data=False
    filtered_data_path = os.path.join(config.CRISIS_DATES,'filtered_data.pkl'.format())
    training_data_path = os.path.join(config.CRISIS_DATES,'train_data.pkl'.format())
    crisis_date_path = os.path.join(config.CRISIS_DATES,'criris_dates.pkl')
    #%%
    if read_data:
        all_country_df= pd.read_pickle(filtered_data_path)
    else:
        all_country_df = pd.DataFrame([])
        for country in config.countries:
            #country='brazil'
            logger.info(country)
            file_path = os.path.join(config.COUNTRY_EMB,'{}_emb.pkl'.format(country))
            if os.path.exists(file_path):
                df= pd.read_pickle(file_path)
                df = filter_country_feature(df,topn=10)
                agg_df = aggregate_emb(df)
                agg_df['country'] = country 
                all_country_df=all_country_df.append(agg_df)
            else:
                logger.warning('{} does not exist'.format(country))
        all_country_df.to_pickle(filtered_data_path)
    #%%
    ##read crisis date
    df_crisis_dates = pd.read_pickle(crisis_date_path)
    #%%
    ## merge data with crisis dates
    train_df= all_country_df.merge(df_crisis_dates,
                                     how='inner', 
                                     left_on=['month','country'],
                                     right_on=['month','transformed_country_name'])
    train_df.to_pickle(training_data_path)