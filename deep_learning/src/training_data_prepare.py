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

def create_load_news_emb(filtered_data_path,read_data=True):
    """
    read and filter news embedings 
    """
    if read_data:
        all_country_df= pd.read_pickle(filtered_data_path)
        logger.info('read from {}'.format(filtered_data_path))
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
        logger.info('saved data to from {}'.format(filtered_data_path))
    return all_country_df


    
#%%
if __name__ == '__main__':
    
    ## set up global arguments
    read_data=True
    window_size = 3
    run_rr = False
    run_kr = True
    run_ve_q = False
    
    crisis_file = os.path.join(config.CRISIS_DATES,'crisis_dates.xlsx')
    country_map = pd.read_excel(crisis_file,'country_code_map')
    ft2ifs_map = dict(zip(country_map.ft_country_name,country_map.ifs_code))    ## create country code map
    ifs2imf_map= dict(zip(country_map.ifs_code,country_map.imf_country_name))   ## create country code map
    
    
    filtered_data_path = os.path.join(config.CRISIS_DATES,'filtered_data.pkl')

    crisis_date_path = os.path.join(config.CRISIS_DATES,'criris_dates.pkl')
    kr_crisis_date_path = os.path.join(config.CRISIS_DATES,'criris_dates_kr_w{}.pkl'.format(window_size))
    ve_q_crisis_date_path = os.path.join(config.CRISIS_DATES,'criris_dates_ve_q.pkl')
    
    training_data_path_rr = os.path.join(config.CRISIS_DATES,'train_data_rr.pkl')
    training_data_path_kr = os.path.join(config.CRISIS_DATES,'train_data_kr_w{}.pkl'.format(window_size))
    training_data_path_ve_q = os.path.join(config.CRISIS_DATES,'train_data_ve_q.pkl')
    #%%
    ###################
    ## read news emb
    all_country_df = create_load_news_emb(filtered_data_path,read_data= read_data)
    all_country_df['ifs_code'] = all_country_df['country'].map(ft2ifs_map)                 ## add country code 
    all_country_df['imf_country_name'] = all_country_df['ifs_code'].map(ifs2imf_map)       ## add country name 
    
    ##############################
    ## merge export training data // rr
    if run_rr:
        df_crisis_dates = pd.read_pickle(crisis_date_path)
        train_df= all_country_df.merge(df_crisis_dates,
                                         how='inner', 
                                         left_on=['month','country'],
                                         right_on=['month','transformed_country_name'])
        train_df.to_pickle(training_data_path_rr)
        logger.info('export to {}'.format(training_data_path_rr))

    ##############################
    ## merge export kr data  // kr data with different window
    if run_kr:
        df_crisis_dates_kr = pd.read_pickle(kr_crisis_date_path)
        train_df= all_country_df.merge(df_crisis_dates_kr,
                                         how='inner', 
                                         left_on=['month','country'],
                                         right_on=['month','country_name'])
        ## export data
        train_df.to_pickle(training_data_path_kr)
        logger.info('export to {}'.format(training_data_path_kr))
    #train_df.drop(['quarter_y','transformed_country_name'],axis=1,inplace=True)

    ##############################
    ## merge export ve data 
    if run_ve_q:
        df_crisis_dates_ve_q = pd.read_pickle(ve_q_crisis_date_path)
        train_df= all_country_df.merge(df_crisis_dates_ve_q,
                                         how='inner', 
                                         left_on=['month','ifs_code'],
                                         right_on=['month','ifscode'])
        train_df.drop(['country','ifscode'],axis=1,inplace=True)
        ## export data
        train_df.to_pickle(training_data_path_ve_q)
        logger.info('export to {}'.format(training_data_path_ve_q))