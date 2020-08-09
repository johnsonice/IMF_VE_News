#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 12:20:00 2020

@author: chuang
"""

## process_crisis_dates_kr


import pandas as pd 
import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os
from process_crisis_dates import transform_criris_data, generate_pre_during_post_crisis_dates

def convert_date(d):
    ## soemthing like ' February 21, 1998, page: 0038'
    date_obj = parse(d)
    #transformed_str = date_obj.strftime("%Y-%m-%d")
    return date_obj

#%%
if __name__ == "__main__":    
    ## get and tansform crisis date
    crisis_file = os.path.join(config.CRISIS_DATES,'crisis_dates.xlsx')
    df = pd.read_excel(crisis_file,sheet_name='kr_crisis')
    df = transform_criris_data(df,convert_date,resample=False)
    ##  
    crisis_df = generate_pre_during_post_crisis_dates(df,shift_periods=24,pid='country_name')
#    crisis_df = crisis_df.rename(columns={'crisisdate':'crisisdate_kr',
#                              'crisis_pre':'criris_pre_kr',
#                              'crisis_tranqull':'crisis_tranqull_kr'})
    ## export data
    crisis_df.to_pickle(os.path.join(config.CRISIS_DATES,'criris_dates_kr.pkl'))
