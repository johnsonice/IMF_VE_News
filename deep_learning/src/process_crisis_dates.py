# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:41:13 2020

@author: CHuang
"""

import pandas as pd 
import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os

#%%

def convert_date(d):
    ## soemthing like ' February 21, 1998, page: 0038'
    clean_date = d.replace(':1','-01-01').replace(':2','-07-01')
    date_obj = parse(clean_date)
    #transformed_str = date_obj.strftime("%Y-%m-%d")

    return date_obj

def transform_criris_data(crisis_df,convert_date=None,resample=True):
    """
    wide format of crisi date to long format
    """
    if convert_date is not None:
        crisis_df['Date'] = crisis_df['Date'].apply(convert_date)
    crisis_df= crisis_df.set_index(['Date'])
    crisis_df.columns = ["crisisdate_"+c for c in crisis_df.columns]
    if resample == True:
        crisis_df= crisis_df.resample('m').pad() # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
    crisis_df['month'] = crisis_df.index.to_period('M')
    crisis_df['quarter'] = crisis_df.index.to_period('Q')
    crisis_df = pd.wide_to_long(crisis_df,stubnames='crisisdate', i = ['month', 'quarter'],j='country_name' ,sep='_', suffix='\w+')
    crisis_df.reset_index(inplace=True)  
    crisis_df = crisis_df.sort_values(by=['country_name','month'])
    
    return crisis_df

def generate_pre_during_post_crisis_dates(crisis_df,shift_periods=12,pid='transformed_country_name'):
    """
    generate pre crisis period and post crisis period 
    """
    
    ##reate label for 1:n years ahead of crisis starts
    df = crisis_df
    df['crisis_pre1'] = df.groupby(pid)['crisisdate'].shift(-1)
    df['crisis_pre1'] = df['crisis_pre1'] - df['crisisdate']
    df['crisis_pre1'].replace(-1,0,inplace=True)
    df['crisis_pre1'].replace(np.nan,0,inplace=True)
    df['crisis_pre'] = df['crisis_pre1']

    for i in range(2,shift_periods+1):
        df['crisis_pre{}'.format(i)] = df.groupby(pid)['crisis_pre1'].shift(-(i-1))
        df['crisis_pre{}'.format(i)].replace(np.nan,0,inplace=True)
        df['crisis_pre'] = df['crisis_pre'] + df['crisis_pre{}'.format(i)]
        df.drop(['crisis_pre{}'.format(i)], axis=1,inplace=True)
    df.drop(['crisis_pre1'], axis=1,inplace=True)
    df['crisis_pre'][df['crisis_pre']>1]=1
    df['crisis_pre'] = df['crisis_pre'] - df['crisisdate']
    df['crisis_pre'][df['crisis_pre']<0]=0
    df['crisis_tranqull'] =  1
    df['crisis_tranqull'] = df['crisis_tranqull'] - df['crisis_pre'] - df['crisisdate']
    return df

#%%
if __name__ == "__main__":    
    ## get and tansform crisis date
    crisis_file = os.path.join(config.CRISIS_DATES,'crisis_dates.xlsx')
    df = pd.read_excel(crisis_file,sheet_name='rr_crisis')
    df = transform_criris_data(df,convert_date)
    
    ## get country map to ft countries
    country_map = pd.read_excel(crisis_file,sheet_name='country_name_map')
    
    ## merge data
    df_final  = df.merge(country_map,
                         how='inner', 
                         left_on='country_name',
                         right_on='Country_name')
    ## clean data
    df_final.drop(['country_name', 'Country_name'], axis=1,inplace=True)
    df_final.dropna(inplace=True)
    
    df_final = generate_pre_during_post_crisis_dates(df_final,24)
    df_final.to_pickle(os.path.join(config.CRISIS_DATES,'criris_dates.pkl'))
    
#    #%%
#    ct_emb = pd.read_pickle('argentina_emb.pkl')
##%%
#    
#    ## need to double check on merging 
#    ct_emb= ct_emb.merge(df_final[['month','crisisdate','formated_country_name']],
#                         how='inner', 
#                         left_on='month',
#                         right_on='month')