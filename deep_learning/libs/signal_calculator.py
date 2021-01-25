#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:13:57 2019

@author: chuang
"""


import pandas as pd
import numpy as np


def rolling_z_score(freqs, window=24):
    def z_func(x):
        return (x[-1] - x[:-1].mean()) / x[:-1].std(ddof=0)
    return freqs.rolling(window=window+1).apply(z_func, raw=True)

def rolling_z_score_pannel(df,panel_id,value_id,window=24):
    """
    calculate rolling z score in pannel setting
    returns a pandas series with the same index as the dataframe passed in 
    """
    def z_func(x):
        return (x[-1] - x[:-1].mean()) / x[:-1].std(ddof=0)
    res = df.groupby(panel_id)[value_id].rolling(window=window+1).apply(z_func,raw=True).reset_index(level=0, drop=True)
    return res 

def signif_change(df,panel_id,value_id,window=24,period='month', direction=None,z_thresh=1.96):
    """
    find periods for which there was a significant change wrt the rolling average.

    freqs: (pd.Series) time series to check
    window: (int) number of periods prior to t over which to calc rolling mean/std/z_score
    direction: (str or NoneType) None for both signif increase and decrease, otherwise 'incr' or 'decr'
    """
    assert isinstance(df, pd.DataFrame)
    fq = period[0].lower()
    assert fq in ['m','q']
    if fq == 'q':
        window = int(window/3)
    
    z_scores = rolling_z_score_pannel(df,panel_id,value_id,window)
    if not direction:
        result = (z_scores >= z_thresh) | (z_scores <= -z_thresh)
    else:
        if 'incr' in direction:
            result = (z_scores >= z_thresh)
        elif 'decr' in direction:
            result = (z_scores <= -z_thresh)
        else: 
            raise ValueError

    return result.astype(int)

def cal_all_signals(df,
                  panel_id='ifscode',
                  window=24,
                  period='month', 
                  direction='incr',
                  z_thresh=2.1):
    
    
    index_columns = [x for x in list(df.columns) if 'index_' in x]
    for inx in index_columns:
        pred_col_name = inx.replace('index_','pred_')
        df[pred_col_name] = signif_change(df,panel_id=panel_id,value_id=inx,
                                          window=24,period=period, 
                                          direction=direction,z_thresh=z_thresh)
    
    return df
    

#%%

if __name__ == '__main__':
    crisis_file = '/data/News_data_raw/FT_WD/crisis_date/crisis_dates.xlsx'
    df = pd.read_excel(crisis_file,sheet_name='news_index_crisis').dropna(subset=['ifscode'])
    
    old_df = df.sort_values(by=['ifscode','time'])
    old_df.fillna(0,inplace=True)
    x = old_df.describe()

    ## recalculate use differnt thresholdthreshold 
    raw_columns = [x for x in list(df.columns) if 'pred_' not in x]    
    df = df[raw_columns].sort_values(by=['ifscode','time'])
    
    df = cal_all_signals(df,
                          panel_id='ifscode',
                          window=24,
                          period='month', 
                          direction='incr',
                          z_thresh=1)  ## specify a different threshold
    
    x = old_df.describe()
    y = df.describe()
    print(x)
    print(y)