#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:00:33 2018

@author: chuang
"""
## add country metadata 
## data exploration, descripitive analysis 

import sys,os
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
#from crisis_points import country_dict
from nltk.tokenize import word_tokenize
from stream import MetaStreamer_fast as MetaStreamer
#import time 
from mp_utils import Mp
import re
country_dict = {
    'argentina': ['argentina'],
    'bolivia': ['bolivia'],
    'brazil': ['brazil'],
    'chile': ['chile'],
    'colombia': ['colombia'],
    'denmark': ['denmark'],
    'finland': ['finland'],
    'indonesia': ['indonesia'],
    'israel': ['israel'],
    'malaysia': ['malaysia'],
    'mexico': ['mexico'],
    'norway': ['norway'],
    'peru': ['peru'],
    'philippines': ['philippines'],
    'spain': ['spain'],
    'sweden': ['sweden'],
    'thailand': ['thailand'],
    'turkey': ['turkey'],
    'uruguay': ['uruguay'],
    'venezuela': ['venezuela'],
    'angola':['angola'],
    'ghana':['ghana'],
    'kenya':['kenya'],
    'mauritius':['mauritius'],
    'mozambique':['mozambique'],
    'nigeria':['nigeria'],
    'senegal':['senegal'],
    'tanzania':['tanzania'],
    'uganda':['uganda'],
    'zambia':['zambia'],
    'zimbabwe':['zimbabwe']
}

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
#def get_country_name(tokens,country_dict):
#    for c,v in country_dict.items():
#        rc = c if tokens and any([tok.lower() in tokens for tok in v]) else None
#        if rc is not None:
#            yield rc 

def construct_rex(keywords):
    r_keywords = [r'\b' + re.escape(k)+ r'(s|es|\'s)?\b' for k in keywords]
    rex = re.compile('|'.join(r_keywords),flags=re.I) ## ignore casing
    return rex

def get_country_name(text,country_dict,rex=None):
    for c,v in country_dict.items():
        rex = construct_rex(v)
        rc = rex.findall(text)
        if len(rc)>0:
            yield c

def get_countries(article,country_dict=country_dict):
    #snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    #title = word_tokenize(article['title'].lower()) if article['title'] else None
    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    if snip and title:
        #title.extend(snip)
        title = "{} {}".format(title,snip)
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
    meta_root = config.DOC_META
    meta_pkl = config.DOC_META_FILE
    json_data_path = config.JSON_LEMMA
    
    ## define add hoc countries to check     
    df = pd.read_pickle(meta_pkl)
    
    #df = df.tail(5000)
    #%%
    
    df['data_path'] = json_data_path+'/'+df.index + '.json'
    print('see one example : \n',df['data_path'].iloc[0])
    streamer = MetaStreamer(df['data_path'].tolist())
    news = streamer.multi_process_files(workers=30,chunk_size=5000)
    #%%
    mp = Mp(news,get_countries)
    country_meta = mp.multi_process_files(workers= 30,chunk_size=5000)
    #%%
    index = [i[0] for i in country_meta]
    country_list = [i[1] for i in country_meta]
    del country_meta ## clear memory
    
    ds = pd.Series(country_list,name='country',index=index)
    df = df.join(ds) ## merge country meta
    df['country_n'] = df['country'].map(lambda x: len(x))
    df.to_pickle(os.path.join(meta_root, 'doc_details_{}_aug.pkl'.format('crisis')))
    print('augumented document meta data saved at {}'.format(meta_root))
    
    #%%
    # create aggregates for ploting
    agg_q = df[['date','quarter']].groupby('quarter').agg('count')
    #agg_m = df[['date','month']].groupby('month').agg('count')
    create_summary(agg_q,meta_root)

