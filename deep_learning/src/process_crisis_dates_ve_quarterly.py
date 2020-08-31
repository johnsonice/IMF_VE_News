#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:05:48 2020

@author: chuang
"""

## process VE quarter dates
import pandas as pd 
import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os
from process_crisis_dates import generate_pre_during_post_crisis_dates


def transform_criris_data(crisis_df,convert_date=None,resample=True):
    """
    wide format of crisi date to long format
    """
    if convert_date is not None:
        crisis_df['date'] = crisis_df['date'].apply(convert_date)
    crisis_df= crisis_df.set_index(['date'])
    if resample == True:
        crisis_df= crisis_df.groupby('ifscode').resample('m').pad()  # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    crisis_df['month'] = crisis_df.index.get_level_values(1).to_period('M')
    crisis_df['quarter'] = crisis_df.index.get_level_values(1).to_period('Q')
    crisis_df.drop(columns=['ifscode'],inplace=True)
    crisis_df.reset_index(inplace=True)  
    crisis_df = crisis_df.sort_values(by=['country','month'])
    
    return crisis_df
#%%
if __name__ == "__main__":    
    ## get and tansform crisis date
    crisis_file = os.path.join(config.CRISIS_DATES,'crisis_dates.xlsx')
    df = pd.read_excel(crisis_file,sheet_name='ve_quarterly_crisis')
    df = df[['ifscode','year','quarter','date','country','crisis_start']]  ## keep columns    
    crisis_df = transform_criris_data(df,resample=True)
    #%%
    ## change variable name to fit fromula 
    crisis_df = crisis_df.rename(columns={
                                      'date':'Date',
                                      'crisis_start':'crisisdate',
                                      'country':'country_name'
                                      })
    #%% create precrisis period
    crisis_df = generate_pre_during_post_crisis_dates(crisis_df,shift_periods=24,pid='country_name')

    ## export data
    crisis_df.to_pickle(os.path.join(config.CRISIS_DATES,'criris_dates_ve_q.pkl'))
