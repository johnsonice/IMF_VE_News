#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:00:33 2018

@author: chuang
"""
## add country metadata 
## data exploration, descripitive analysis 

import sys,os
sys.path.insert(0,'./libs')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from crisis_points import country_dict
from nltk.tokenize import word_tokenize
from stream import MetaStreamer_fast as MetaStreamer
#import time 
from mp_utils import Mp
#plt.rcParams['figure.figsize']=(10,5)

#%%
## save quarterly figure
def create_summary(agg_q,meta_root):
    x_ticker = agg_q.index[0::4]
    ax = agg_q.plot(figsize=(16,6),title='News Articles Frequency',legend=False)
    plt.ylabel('Number of news articles')       ## you can also use plt functions 
    plt.xlabel('Time-Q') 
    ax.set_xticks(x_ticker)                     ## set ticker
    ax.set_xticklabels(x_ticker,rotation=90)    ## set ticker labels
    ax.get_xaxis().set_visible(True)            ## some futrther control over axis
    plt.savefig(os.path.join(meta_root,'quarter_summary.png'),bbox_inches='tight')

#%%
def get_country_name(tokens,country_dict):
    for c,v in country_dict.items():
        rc = c if tokens and any([tok.lower() in tokens for tok in v]) else None
        if rc is not None:
            yield rc 

def get_countries(article,country_dict=country_dict):
    snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    title = word_tokenize(article['title'].lower()) if article['title'] else None

    if snip and title:
        title.extend(snip)
        cl = list(get_country_name(title,country_dict))
    elif title:
        cl = list(get_country_name(title,country_dict))
    elif snip:
        cl = list(get_country_name(snip,country_dict))
    else:
        cl = list()
        
    return article['an'],cl    

#%%
if __name__ == '__main__':
    meta_root = "/data/News_data_raw/FT_WD/doc_meta"
    meta_pkl = os.path.join(meta_root, "doc_details_crisis.pkl")
    json_data_path = '/data/News_data_raw/Financial_Times_processed/FT_json_historical/'
    
    df = pd.read_pickle(meta_pkl)
    #%%
    
    #df= df.head(5000)
    df['data_path'] = json_data_path +df.index + '.json'
    print('see one example : \n',df['data_path'].iloc[0])
    streamer = MetaStreamer(df['data_path'].tolist())
    news = streamer.multi_process_files(workers=31,chunk_size=5000)
    #%%
    #country_meta = [(a['an'],get_countries(a,country_dict)) for a in news]
    mp = Mp(news,get_countries)
    country_meta = mp.multi_process_files(workers= 31,chunk_size=5000)
    #%%
    index = [i[0] for i in country_meta]
    country_list = [i[1] for i in country_meta]
    del country_meta ## clear memory
    
    ds = pd.Series(country_list,name='country',index=index)
    df = df.join(ds) ## merge country meta
    df['country_n'] = df['country'].map(lambda x: len(x))
    df.to_pickle(os.path.join(meta_root, 'doc_details_{}_aug.pkl'.format('crisis')))
    
    #%%
    # create aggregates for ploting
    agg_q = df[['date','quarter']].groupby('quarter').agg('count')
    #agg_m = df[['date','month']].groupby('month').agg('count')
    create_summary(agg_q,meta_root)

