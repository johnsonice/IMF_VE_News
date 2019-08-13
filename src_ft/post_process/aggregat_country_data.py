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
    
sys.path.insert(0,os.path.join(cwd,'../libs'))
sys.path.insert(0,os.path.join(cwd,'..'))
import argparse
import config 
import re 
import pandas as pd
#from tooltip_generator import Tool_tips_generator 
#from infer_utils import get_current_date
import datetime as dt
## some inference utility functions 

#%%
def maybe_create(f_path):
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        print('Generate folder : {}'.format(f_path))
    return None

def get_current_date():
    res = dt.datetime.strftime(dt.datetime.today(),"%Y-%m-%d")
    return res

def get_current_month():
    res = dt.datetime.strftime(dt.datetime.today(),"%Y-%m")
    return res


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
    merge_df.to_csv(os.path.join(output_path,'contry_data_wide_{}.csv'.format(get_current_date())))
    
    new_df_long= pd.wide_to_long(df=merge_df,stubnames=['index','pred'],i=['time','country_name'],j='indexes',sep='-',suffix='\\w+')
    new_df_long.reset_index(inplace=True)
    new_df_long.to_csv(os.path.join(output_path,'contry_data_long_{}.csv'.format(get_current_date())))
    
    
    return merge_df,new_df_long
    ## merge tooltips  
    #TG = Tool_tips_generator(config)
    #df_tt = TG.get_tool_tips_df(topn=3) ## tool tip df
    
    ## overwrite historical df_tt
    #df_tt.to_pickle(os.path.join(config.HISTORICAL_INPUT,'tool_tips_df.pkl'))
    
#    ## prepare df_tt for merge
#    df_tt['country_name'] = df_tt['country_name'].apply(lambda x:x.title())
#    df_tt['time'] = df_tt['time'].apply(lambda x:str(x))
#    new_df = pd.merge(new_df,df_tt,how='left',on=['time','country_name','indexes'])
#
#    ## clean up and export     
#    new_df['indexes'] = new_df['indexes'].apply(lambda s:var_name_map[s])
#    if export:
#        new_df.to_csv(os.path.join(output_path,'contry_data_long_{}.csv'.format(get_current_date())))
#    
#    return new_df

#%%

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts_path', '--ts_path', action='store', dest='ts_path', 
                        default=config.EVAL_TS)
    parser.add_argument('-out_dir', '--out_dir', action='store', dest='out_dir', 
                        default='./temp_data')
    
    args = parser.parse_args()
    #args.ts_path=config.HISTORICAL_TS_PS
    #%%
    new_df,df_long = export_tableau_data(args.ts_path,args.out_dir)



