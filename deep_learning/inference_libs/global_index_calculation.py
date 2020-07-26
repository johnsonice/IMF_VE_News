# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 08:59:46 2019

@author: chuang
"""
import pandas as pd
import numpy as np
import copy
#%%

def _calculate_global_index(df,weights_path='global_index_weights.xlsx'):
    ## step 0, define keep variables 
    df = copy.copy(df)
    keep_vars = ['time','country_name','indexes','index']
    
    ## step 1, get country weights based on pppgdp share
    df_weights = pd.read_excel(weights_path,'global_index_weights')
    weights_dict = dict(zip(df_weights['News_countries'],df_weights['PPPGDP_weights']))
    
    ## step 2 generate global index -- normalize
    #df = df[df['indexes']=='All sentiment']
    df.sort_values(by=['indexes','country_name','time'],inplace=True)
    df['temp_index'] = df['index'].replace(0, np.nan).bfill()
    df['base'] = df.groupby(['country_name','indexes'])['temp_index'].transform('first')
    df['index_norm'] = df['index']/df['base']*100
    
    ## step 2 generate global index -- aggregate by weighted sum
    df['weights'] = df['country_name'].map(weights_dict)
    df['index_norm_weighted'] = df['weights']*df['index_norm']
    global_index_df = df.groupby(['time','indexes'],as_index=False).sum()
    
    ## step 3 do some clean ups 
    global_index_df['index'] = global_index_df['index_norm_weighted']
    global_index_df['country_name'] = 'World' 
    global_index_df = global_index_df[keep_vars]
    
    return global_index_df

def _append_global(df_master,df_global):
    df_final = df_master.append(df_global,ignore_index=True)
    return df_final

def get_merge_global_index(df,weights_path='global_index_weights.xlsx'):
    global_df = _calculate_global_index(df,weights_path)
    final_df = _append_global(df,global_df)
    return final_df
    
    #%%
if __name__ == "__main__":
    path = 'contry_data_long.csv'
    df = pd.read_csv(path)#[keep_vars]
    global_df = _calculate_global_index(df)
    final_df=_append_global(df,global_df)
    #global_df.set_index('time',inplace=True)

    global_df[global_df['indexes']=='Crisis sentiment'].plot(x='time',y='index',figsize=(10,4))
    #global_df.to_csv('global.csv',index=False)
    