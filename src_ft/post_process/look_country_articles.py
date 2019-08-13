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
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np
from multiprocessing import cpu_count, Pool
from mp_utils import Mp
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

def get_df_for_period(period_start,period_end,df,freq='month'):
    df_period = df[(df[freq]>=period_start)&(df[freq]<=period_end)]
    #df_period = df_period[df_period[freq]<=period_end]
    return df_period

def writeout_sample_article(link,output_file_path):
    with open(link, 'r', encoding="utf-8") as f:
        data = json.loads(f.read())
        
    
    with open(output_file_path,'w',encoding='utf-8') as of:
        of.write(data['body'])
    
    return None

def read_json(l):
    try:
        with open(l, 'r', encoding="utf-8") as f:
            dj = json.loads(f.read())
        return dj
    except:
        return None

def trans_article_to_df(links,multi=False):
    df = pd.DataFrame()

    if multi:
        mpor = Mp(links,read_json)
        data = mpor.multi_process_files(workers=25,chunk_size=200)
#        for d in data:
#            try:
#                df = df.append(d,ignore_index=True)
#            except:
#                pass
        df = pd.DataFrame(data)
    else:
        for l in links:
#            with open(l, 'r', encoding="utf-8") as f:
#                data = json.loads(f.read())
#                #print(data['title'])
            data = read_json(l)
            try:
                df = df.append(data,ignore_index=True)
            except:
                pass

    return df

def get_input_words_weights(topn,wg,vecs=None,weighted=False):
    # use topn most similar terms as words for aggregate freq if args.sims

    try:
        sims = [w for w in vecs.wv.most_similar(wg, topn=topn)]    ## get similar words and weights 
    except KeyError:
        try:
            print('{} was splited for sim words searching..'.format(wg))
            wg_update = list()
            for w in wg:
                wg_update.extend(w.split('_'))
            sims = [w for w in vecs.wv.most_similar(wg_update, topn=topn)]
        except:
            #print('Not in vocabulary: {}'.format(wg_update))
            raise Exception('Not in vocabulary: {}'.format(wg_update))
            
    wgw = [(w,1) for w in wg]  ## assign weight 1 for original words
    words_weights = sims + wgw   
    
    ## get words and weights as seperate list
    words = [w[0] for w in words_weights]
    
    if weighted:    
        weights = [w[1] for w in words_weights]
    else:
        weights= None
    
    return words,weights

def read_grouped_search_words(file_path):
    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k,v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]
        temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
        search_groups[k]=temp_list
    return search_groups

def get_sim_words_set(topn,word_group,vecs):
    assert isinstance(word_group,list)     
    sim_word_group = list()
    for w in word_group:
        words, weights = get_input_words_weights(topn,
                                             w,
                                             vecs=vecs,
                                             weighted=False)
        sim_word_group.extend(words)
    sim_word_set = set(sim_word_group)
    return sim_word_set



def get_counts(body,search_sets):
    assert isinstance(body,str) 
    token_list = word_tokenize(body)
    token_counter = Counter(token_list)
    res = 0 
    for i in search_sets:
        res += token_counter[i]
    return res

def paralleize(data,func,cores,partitions):
    data_split = np.array_split(data,partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func,data_split))
    pool.close()
    pool.join()
    return data


#%%

if __name__ == '__main__':
    country = 'united-states'
    month_start = '1980-01'
    month_end = '2017-12'
    out_folder = './temp_data/country_articles'

#%%
    config.countries = ['united-kingdom']
    for country in config.countries:
        print('working on country {}'.format(country))
        # read meta data and get file links
        df = read_meta()
        df_country = get_country_df(country,df)
        df_country_period = get_df_for_period(month_start,month_end,df_country,freq='month')
    
        links = df_country_period['data_path'].values.tolist()
        print('number of documents to process: {}'.format(len(links)) )
    
    #    for idx,l in enumerate(links):
    #        writeout_sample_article(l,os.path.join(out_folder,'{}.txt'.format(idx)))
    
    #%%
        article_df = trans_article_to_df(links,multi=True)
        article_df= article_df[['publication_date','title','snippet','body']]
        #article_df.to_excel(os.path.join(out_folder,'{}_{}.xlsx'.format(country,month)))
        
    #    with open(links[5], 'r', encoding="utf-8") as f:
    #        data = json.loads(f.read())
    #        print(data['title'])
    #    #%%
    #    td = pd.DataFrame.from_dict(data,orient='index')
        
    #%%
        # load wv and generate documetn states
        vecs = KeyedVectors.load(config.W2V)
        org_word_group = read_grouped_search_words(os.path.join(config.SEARCH_TERMS,config.GROUPED_SEARCH_FILE))
        search_words_sets = {k:get_sim_words_set(config.topn,v,vecs) for k,v in org_word_group.items()}
        
        def produce_document_stats(df,search_words_sets=search_words_sets):
            for k,v in search_words_sets.items():
                df[k] = df['body'].apply(get_counts,args=(v,))
            
            return df 
        
        out_df = paralleize(article_df,produce_document_stats,cores=24,partitions=48)
        out_df['month'] = out_df['publication_date'].apply(lambda x: x[:7])
        #out_df = produce_document_stats(article_df,search_words_sets)
        if len(out_df)>100000:
            pass
            #out_df.to_csv(os.path.join(out_folder,'{}.csv'.format(country)))
        else:
            out_df.to_excel(os.path.join(out_folder,'{}.xlsx'.format(country)))
        #%%
        # only top 20 in each group 
        out_df.sort_values(by=['month','all_language'],ascending=False,inplace=True)
        out_df = out_df.groupby('month').head(20)
        #%%
        try:
            out_df.to_excel(os.path.join(out_folder,'{}_top20.xlsx'.format(country)))
        except:
            out_df.to_csv(os.path.join(out_folder,'{}_top20.csv'.format(country)))
    