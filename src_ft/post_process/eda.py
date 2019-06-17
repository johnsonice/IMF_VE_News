#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:54:42 2019

@author: chuang
"""

import sys,os
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config
import pandas as pd
from datetime import datetime as dt
#%%

df = pd.read_pickle(config.AUG_DOC_META_FILE)
#%%
agg_m= df[['date','month']].groupby('month').agg('count')
agg_m = agg_m[:'2017-12']
#%%
print(agg_m.sum()) 
print(agg_m.mean()) 
#%%

def get_country_df(country_name,df):
    fl = df['country'].apply(lambda x: country_name in x)
    df_c = df[fl]
    agg_df_c = df_c[['date','month']].groupby('month').agg('count')
    agg_df_c.columns =[country_name]
    return agg_df_c,df_c
x,y = get_country_df('united-kingdom',df)
#%%
country_df_list = [get_country_df(c,df) for c in config.countries]
country_agg_df = pd.concat(country_df_list,axis=1)

#%%
country_agg_df.mean()

#%%
export_file = os.path.join(config.DOC_META,'meta_summary_0617.xlsx')
with pd.ExcelWriter(export_file) as writer:
    agg_m.to_excel(writer,sheet_name='overall')
    country_agg_df.to_excel(writer,sheet_name='country_level')
    