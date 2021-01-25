#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:05:48 2020

@author: chuang
"""

## process VE quarter dates
import pandas as pd 
#import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os
#from process_crisis_dates import generate_pre_during_post_crisis_dates
from signal_calculator import cal_all_signals


def convert_date(d):
    ## soemthing like ' February 21, 1998, page: 0038'
    date_obj = parse(d)
    #transformed_str = date_obj.strftime("%Y-%m-%d")
    return date_obj

def prepare_news_index(df):
    
    ## pre process news index, create signal dummy
    signal_columns = [c for c in df.columns if 'pred_' in c]
    df['pred_crisis_agg'] = df[signal_columns].sum(axis = 1) ## row sum of all signals 
    df['crisisdate_news'] = df['pred_crisis_agg'].apply(lambda i: 1 if i >0 else 0)
    df = df[['time','imf_country_name', 'ifscode','pred_crisis_agg','crisisdate_news']]
    df = df.rename(columns={
                          'time':'Date',
                          'imf_country_name':'country_name'
                          })
    return df

def transform_criris_data(crisis_df,convert_date=None,resample=True):
    """
    wide format of crisi date to long format
    """
    if convert_date is not None:
        crisis_df['Date'] = crisis_df['Date'].apply(convert_date)
    crisis_df= crisis_df.set_index(['Date'])
    if resample == True:
        crisis_df= crisis_df.groupby('ifscode').resample('m').pad()  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    crisis_df['month'] = crisis_df.index.get_level_values(0).to_period('M')
    crisis_df['quarter'] = crisis_df.index.get_level_values(0).to_period('Q')
    #crisis_df.drop(columns=['ifscode'],inplace=True)
    crisis_df.reset_index(inplace=True)  
    crisis_df = crisis_df.sort_values(by=['country_name','month'])
    
    return crisis_df
#%%
if __name__ == "__main__":    
    
    ## specify signal threshold you want to use 

    z_threshold = 1 ## or None
    
    ## get and tansform crisis date
    crisis_file = os.path.join(config.CRISIS_DATES,'crisis_dates.xlsx')
    df = pd.read_excel(crisis_file,sheet_name='news_index_crisis')

    ## recalculate use differnt threshold
    raw_columns = [x for x in list(df.columns) if 'pred_' not in x]    
    df = df[raw_columns].sort_values(by=['ifscode','time'])
    df = cal_all_signals(df,
                          panel_id='ifscode',
                          window=24,
                          period='month', 
                          direction='incr',
                          z_thresh=z_threshold)  ## specify a different threshold

    ## data cleaning, aggregation
    df = prepare_news_index(df)
    ## create time index etc
    crisis_df = transform_criris_data(df,convert_date=convert_date,resample=False)
    ## export data
    crisis_df.to_pickle(os.path.join(config.CRISIS_DATES,'criris_dates_news_index.pkl'))
