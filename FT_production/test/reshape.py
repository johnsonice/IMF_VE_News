#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 08:54:49 2019

@author: chuang
"""

### reshape long to wide 
import os, sys
sys.path.insert(0,'../inference_libs')
import infer_config as config
import pandas as pd


def long_to_wide(df):
    keep_vars = ['time','country_name','indexes','index','pred']
    df = df[keep_vars]
    temp_df = pd.pivot_table(df,values = ['index','pred'],index=['time','country_name'],columns='indexes')
    temp_df = temp_df.reset_index()
    var_s = ['_'.join(col).strip('_').replace(" ","_") for col in temp_df.columns]
    
    temp_df.columns = var_s
    
    return temp_df

#%%
out_dir = os.path.join(config.PROCESSING_FOLDER,'data','final_results')
file_path = os.path.join(out_dir,'contry_data_long_2019-09-03.csv')
df = pd.read_csv(file_path)
df_wide = long_to_wide(df)
out_path = os.path.join(out_dir,'contry_data_wide_2019-09-03.csv')
df_wide.to_csv(out_path,index=False)


