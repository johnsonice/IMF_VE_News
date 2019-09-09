#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:53:17 2019

@author: chuang
"""

import os
import sys
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
    
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))
import argparse
import infer_config as config 
import re 
import pandas as pd
from tooltip_generator import Tool_tips_generator 
from global_index_calculation import get_merge_global_index
from infer_utils import get_current_date

#%%

def get_country_name(file_name):
    cn = re.search(r'agg_(.*?)_month',file_name).group(1)
    return cn.replace("_"," ").title()

#test = 'agg_argentina_month_z2.1_time_series.csv'
#print(get_country_name(test))

def get_country_df(df_path):
    cn = get_country_name(df_path)
    df = pd.read_csv(df_path)
    df['country_name'] = cn
    return df

#test = file_pathes[0]
#df = get_country_df(test)

def aggregate_all_countries(data_path):
    file_pathes = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    for idx,fs in enumerate(file_pathes):
        if idx == 0 :
            m_df = get_country_df(fs)
        else:
            c_df = get_country_df(fs)
            m_df = m_df.append(c_df,ignore_index = True)
    
    ## rename first variable to time 
    m_df.columns.values[0] = 'time'
    return m_df

def rename_column_rule(c_name):
    if c_name in ['time','country_name']:
        return c_name
    elif '_pred_crisis' in c_name:
        return 'pred-'+c_name.replace('_pred_crisis','')
    else:
        return 'index-'+c_name
    
    
#test = config.CURRENT_TS_PS
#df = aggregate_all_countries(test)
def export_tableau_data(ts_path,output_path):
    
    var_name_map={
        'agg_all_other_sentiments':'All sentiment (w/o pos and neg)',
        'agg_other_and_negative':'Negative sentiment +',
        'all_language':'All sentiment',
        'crisis_language':'Crisis sentiment',
        'fear_language': 'Fear sentiment',
        'hedging_language': 'Hedging sentiment',
        'negative_sentiment_language': 'Negative sentiment',
        'opinion_language': 'Opinion sentiment',
        'positive_sentiment_language':'Positive sentiment',
        'risk_language':'Risk sentiment'
        }

    merge_df = aggregate_all_countries(ts_path)
    #vns = [f.replace('_pred_crisis',"") for f in merge_df.columns.values if '_pred_crisis' in f]
    new_vns = [rename_column_rule(f) for f in merge_df.columns.values]
    merge_df.columns.values[:] = new_vns
    keep_names = [n for n in new_vns if n not in ['pred-crisis_window','pred-bop_crisis','index-crisis_window','index-bop_crisis']]
    keep_names = ['time','country_name'] + keep_names[1:-1]
    merge_df=merge_df[keep_names]
    
    new_df = pd.wide_to_long(df=merge_df,stubnames=['index','pred'],i=['time','country_name'],j='indexes',sep='-',suffix='\\w+')
    new_df.reset_index(inplace=True)
    
    ## merge tooltips  
    TG = Tool_tips_generator(config)
    df_tt = TG.get_tool_tips_df(topn=3) ## tool tip df
    
    ## overwrite historical df_tt
    df_tt.to_pickle(os.path.join(config.HISTORICAL_INPUT,'tool_tips_df.pkl'))
    
    ## prepare df_tt for merge
    df_tt['country_name'] = df_tt['country_name'].apply(lambda x:x.title())
    df_tt['time'] = df_tt['time'].apply(lambda x:str(x))
    new_df = pd.merge(new_df,df_tt,how='left',on=['time','country_name','indexes'])

    ## clean up and export     
    new_df['indexes'] = new_df['indexes'].apply(lambda s:var_name_map[s])
    ## calculate and merge global index
    new_df = get_merge_global_index(new_df,weights_path=config.INDEX_WEIGHTS)
    ## export to file 
    new_df.to_csv(os.path.join(output_path,'country_data_long_{}.csv'.format(get_current_date())))
    new_df.to_excel(os.path.join(output_path,'country_data_long_{}.xlsx'.format(get_current_date())))
    
    return new_df

def long_to_wide(df):
    """
    reshape long data to wide data for people to download
    """
    keep_vars = ['time','country_name','indexes','index','pred']
    df = df[keep_vars]
    temp_df = pd.pivot_table(df,values = ['index','pred'],index=['time','country_name'],columns='indexes')
    temp_df = temp_df.reset_index()
    var_s = ['_'.join(col).strip('_').replace(" ","_") for col in temp_df.columns]
    
    temp_df.columns = var_s
    
    return temp_df
#%%

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts_path', '--ts_path', action='store', dest='ts_path', 
                        default=config.CURRENT_TS_PS)
    parser.add_argument('-out_dir', '--out_dir', action='store', dest='out_dir', 
                        default=os.path.join(config.PROCESSING_FOLDER,'data'))
    
    args = parser.parse_args()
    #args.ts_path=config.HISTORICAL_TS_PS
    new_df = export_tableau_data(args.ts_path,args.out_dir)



