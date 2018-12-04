# -*- coding: utf-8 -*-

import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
from stream import MetaStreamer_fast as MetaStreamer
import ujson as json
import config


#%%

def country_period_filter(time_df,country,period):
    
    time_df['filter_country'] = time_df['country'].apply(lambda c: country in c)
    df = time_df['data_path'][(time_df['filter_country'] == True)&(time_df['quarter'] == period)]
    
    return df.tolist()
    
#%%
file_path = config.AUG_DOC_META_FILE
time_df = pd.read_pickle(file_path)

#%%
test_t = '1980Q1'
t = pd.to_datetime(test_t).to_period("Q")
c = 'argentina'

#%%
doc_list = country_period_filter(time_df,c,t)

#%%
with open(doc_list[0], 'r', encoding="utf-8") as f:
    data = json.loads(f.read())

print(data['body'])