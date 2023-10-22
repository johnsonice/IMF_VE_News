# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:30:42 2023

Check download files and see if there are large amount of missings 

"""
#%%
import os,sys
sys.path.insert(0, './libs')
from utils import get_all_files,read_json,list_difference_left,retry
import pandas as pd
##newpaper docs
##https://newspaper.readthedocs.io/en/latest/user_guide/advanced.html#parameters-and-configurations
import base64,re
from tqdm import tqdm
import json
import time 
import os, sys , ssl


def count_none(series):
    return (series.isnull().sum())
#%%
if __name__ == "__main__":
    raw_data_f = '/data/chuang/news_scrape/data/raw_try1'
    res_data_f = '/data/chuang/news_scrape/data/news_with_body_try1'
    news_agency = None # none means get them all 'thestkittsnevisobserver'
    ## get remaining files to download 
    file_ps = get_all_files(raw_data_f,end_with='.json',start_with=news_agency,return_name=True) ## cnbc bloomberg reuters
    downlaoded_ps = get_all_files(res_data_f,end_with='.json',start_with=news_agency,return_name=True)
    #%%
    download_stats=[]
    for f in file_ps:
        news,time_start,time_end = f.split("_")
        time_end= time_end.split('.')[0]
        file_p = os.path.join(raw_data_f,f)
        data = read_json(file_p)
        j_name = os.path.basename(file_p)
        download_stats.append((news,time_start,time_end,len(data)))
        df = pd.DataFrame(download_stats)
    ### inspect the df see the times series, see if there are problems 
    #%%

    download_body_stats=[]
    for f in downlaoded_ps:
        news,time_start,time_end = f.split("_")
        time_end= time_end.split('.')[0]
        file_p = os.path.join(res_data_f,f)
        data = read_json(file_p)
        for d in data:
            year,month,day = d.get('published_parsed')[:3]
            temp_res = (news,time_start,time_end,year,month,day,d.get('title'))
            ext = d.get('extracted_content')
            if ext:
                if ext.get('has_body'):
                    news_body = ext.get('bodyXML')
                else:
                    news_body = None
            temp_res = temp_res + (news_body,)
            download_body_stats.append(temp_res)
        
    df = pd.DataFrame(download_body_stats)
    #%%
    df.columns = ['news','start','end','year','month','day','title','body']
    # %%
    agg_operations = {
        'body': [count_none], 
        'news': ['count'] # Aggregations for Value1 column
    }
    #missing_report = df.groupby([0,3,4])[7].apply(percent_none).reset_index()
    missing_report = df.groupby(['news','year','month']).agg(agg_operations).reset_index()
    #%%
    missing_report.columns = ['_'.join(col).strip() for col in missing_report.columns.values]

# %%
