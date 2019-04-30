#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:03:32 2019

@author: chuang
"""

import sys,os
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config
import pandas as pd
#from datetime import datetime as dt
import ujson as json

#%%

def read_meta(meta_path = config.AUG_DOC_META_FILE):
    df = pd.read_pickle(meta_path)
    return df

def get_country_df(country_name,df,agg=False):
    fl = df['country'].apply(lambda x: country_name in x)
    df_c = df[fl]
    if agg:
        agg_df_c = df_c[['date','month']].groupby('month').agg('count')
        agg_df_c.columns =[country_name]
        return agg_df_c
    return df_c

def get_df_for_period(period,df,freq='month'):
    df_period = df[df[freq]==period]
    return df_period

def writeout_sample_article(link,output_file_path):
    with open(link, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
        
    
    with open(output_file_path,'w',encoding='utf-8') as of:
        of.write(data['body'])
    
    return None

def trans_article_to_df(links):
    df = pd.DataFrame()
    for l in links:
        with open(l, 'r', encoding="utf-8") as f:
            data = json.loads(f.read())
            #print(data['title'])
        df = df.append(data,ignore_index=True)

    return df
#%%

if __name__ == '__main__':
    country = 'mexico'
    month = '2019-04'
    out_folder = './temp_data/country_articles'

    df = read_meta()
    df_country = get_country_df(country,df)
    df_country_period = get_df_for_period(month,df_country,freq='month')

    links = df_country_period['data_path'].values.tolist()

    for idx,l in enumerate(links):
        writeout_sample_article(l,os.path.join(out_folder,'{}.txt'.format(idx)))

#%%
    article_df = trans_article_to_df(links)
    article_df= article_df[['publication_date','title','snippet','body']]
    article_df.to_excel(os.path.join(out_folder,'{}.xlsx'.format(country)))
    
#    with open(links[5], 'r', encoding="utf-8") as f:
#        data = json.loads(f.read())
#        print(data['title'])
#    #%%
#    td = pd.DataFrame.from_dict(data,orient='index')