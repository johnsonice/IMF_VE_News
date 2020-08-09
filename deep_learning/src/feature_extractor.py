#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 22:21:44 2020

@author: chuang
"""
import logging
import os,sys
import config
import argparse
sys.path.insert(0,'../libs')
import pandas as pd
import copy
import ujson as json
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter
#from nltk.tokenize import word_tokenize
#import numpy as np
#from multiprocessing import cpu_count, Pool
from mp_utils import Mp
from bert_encoder import baseline_model
import logging
logging.basicConfig(level=logging.ERROR)
logger=logging.getLogger(__name__)

def read_meta(meta_path):
    df = pd.read_pickle(meta_path)
    return df

def get_country_df(country_name,df,period_start=None,period_end=None,freq='month',datafolder=None,agg=False):
    fl = df['country'].apply(lambda x: country_name in x)
    df_c = df[fl]
    
    if period_end is not None:
        df_c = df_c[df_c[freq]<=period_end]
        
    if period_start is not None:
        df_c= df_c[df_c[freq]>=period_start]

    if  datafolder is not None:
        df_c['data_path'] = df_c['data_path'].apply(lambda p:os.path.join(datafolder,os.path.basename(p)))
    
    if agg:
        agg_df_c = df_c[['date','month']].groupby('month').agg('count')
        agg_df_c.columns =[country_name]
        return agg_df_c
    
    return df_c

def read_json(l):
    try:
        with open(l, 'r', encoding="utf-8") as f:
            dj = json.loads(f.read())
        return dj
    except:
        return None

def batch_read_json(links,multi=False):
    #df = pd.DataFrame()
    if multi:
        mpor = Mp(links,read_json)
        data = mpor.multi_process_files(workers=25,chunk_size=200)
#        df = pd.DataFrame(data)
    else:
        data = []
        for l in links:
            data.append(read_json(l))
#            try:
#                df = df.append(data,ignore_index=True)
#            except:
#                pass
    return data

def get_input_words_weights(wg,vecs=None,sims=True,topn=20,weighted=False):
    # use topn most similar terms as words for aggregate freq if args.sims
    if sims:
        #vecs = KeyedVectors.load(args.wv_path)
        if vecs is None:
            raise Exception('w2v model need to be passed in get get similary words.')
        
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
    # otherwise the aggregate freq is just based on the term(s) in the current wg.
    else:
        wgw = [(w,1) for w in wg]  ## assign weight 1 for original words
        words_weights = wgw
    
    ## get words and weights as seperate list
    words = [w[0] for w in words_weights]
    
    if weighted:    
        weights = [w[1] for w in words_weights]
    else:
        weights= None
    
    return words,weights

class doc_processor(object):
    
    def __init__(self,search_words_path=None,search_words_sets=None,w2v_path=None,topn=20,bert_path=None):
        
        
        self.search_words_sets = search_words_sets
        self.topn=topn
        if w2v_path:
            self.vecs = KeyedVectors.load(w2v_path)
        if search_words_path:
            self.search_words_sets=self._get_search_words_sets(search_words_path,topn)
        if bert_path:
            self.model = baseline_model(bert_path)
        print('initiate document processor')
        
    @staticmethod
    def _read_grouped_search_words(file_path):
        df = pd.read_csv(file_path)
        search_groups = df.to_dict()
        for k,v in search_groups.items():
            temp_list = [i for i in list(v.values()) if not pd.isna(i)]
            temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
            search_groups[k]=temp_list
        return search_groups
    
    def _get_sim_words_set(self,word_group,topn):
        assert isinstance(word_group,list)     
        sim_word_group = list()
        for w in word_group:
            words, weights = get_input_words_weights(w,
                                                     vecs=self.vecs,
                                                     sims=True,
                                                     weighted=False,
                                                     topn=topn)
            sim_word_group.extend(words)
        sim_word_set = set(sim_word_group)
        return sim_word_set
    
    def _get_search_words_sets(self,search_words_path,topn=20):
        #file_path = os.path.join(config.SEARCH_TERMS,config.GROUPED_SEARCH_FILE)  ## searh words name
        search_groups = self._read_grouped_search_words(search_words_path) 
        search_words_sets = dict()
        for k,v in search_groups.items():
            search_words_sets[k]=list(self._get_sim_words_set(search_groups[k],topn)) ## turn set to list

        return search_words_sets
    
    def get_counts(self,token_list,search_sets):
        token_counter = Counter(token_list)
        res = 0 
        for i in search_sets:
            res += token_counter[i]
        return res
    
    def produce_document_stats(self,content,search_words_sets=None):
        
        stats = {}
        token_list = content.split()
        if  search_words_sets is None and self.search_words_sets is not None:
            search_words_sets = self.search_words_sets
            
        for k,v in search_words_sets.items():
            stats[k] =self.get_counts(token_list,v)
        
        return stats
    
    def process_file(self,doc):
        if doc['body'] is None:
            doc['body'] = ""
        snip = doc['snippet'].lower() if doc['snippet'] else doc['body']
        title = doc['title'].lower() if doc['title'] else None
        
        stats = self.produce_document_stats(doc['body'])
        stats['snip'] = snip
        stats['title'] = title
            
        return stats
    
    def process_from_json(self,json_path):
        doc = read_json(json_path)
        if doc is None:
            print(json_path)
            return None
        else:
            res = self.process_file(doc)
            return res
    
    def get_emb_from_dict(self,content_dict, k):
        try:
            return self.model.bert_encode2(content_dict[k])[0]
        except:
            logger.error(content_dict)
            return None
#%%
if __name__ == '__main__':
#    try:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--countries', nargs='+', help='countries to get freq for',
                        default=config.countries)
    parser.add_argument('-corp', '--corpus', action='store', dest='corpus', 
                        default=config.JSON_LEMMA)
    parser.add_argument('-doc_meta', '--doc_details', action='store', dest='doc_meta', 
                        default=config.AUG_DOC_META_FILE)
    parser.add_argument('-p', '--period', action='store', dest='period', 
                        default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-s', '--save_dir', action='store', dest='out_dir', 
                        default=config.FREQUENCY)
    parser.add_argument('-ph', '--phraser', action='store', dest='phraser', 
                        default=config.PHRASER)
    parser.add_argument('-kw', '--keywords_path', action='store', dest='keywords_path', 
                    default=os.path.join(config.SEARCH_TERMS,config.GROUPED_SEARCH_FILE))
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', default=config.W2V)
    parser.add_argument('-f', '--wv_filter', action='store', dest='wv_filter', default='TRUE')
    parser.add_argument('-tp', '--topn', action='store', dest='topn', type=int,default=config.topn)
    parser.add_argument('-bp', '--bert_path', action='store', dest='bert_path',default=config.BERT_MODEL)
    parser.add_argument('-do_feature', '--do_feature', action='store', dest='do_feature',default="0")
    parser.add_argument('-do_embed', '--do_embed', action='store', dest='do_embed',default="1")
    args = parser.parse_args()
    
    #%%
    ## add hoc hcange on country groups 
    start_country= 'south-africa'
    args.countries = args.countries[args.countries.index(start_country):]
    
    
    #%%
    ## initiate DP
    DP= doc_processor(search_words_path=args.keywords_path,
                      w2v_path=args.wv_path,
                      topn=args.topn,
                      bert_path=args.bert_path)
    #%%
    ## read document meta data 
    time_df = pd.read_pickle(args.doc_meta)
    uniq_periods = set(time_df[args.period])
    time_df = time_df[time_df['country_n']>0]   
    
    #%%
    if args.do_feature == "1":
        print("... process bag of words features ...")
        for c in args.countries:
            print(".... working on .... {}".format(c))
            country_df = get_country_df(c,time_df)
            country_df['features'] = country_df['data_path'].apply(DP.process_from_json)
            out_path = os.path.join(config.COUNTRY_EMB,"{}.pkl".format(c))
            country_df.to_pickle(out_path)
    #%%
    if args.do_embed == "1":
        print("... process embeding features ...")
        for c in args.countries:
            print(".... working on embeding .... {}".format(c))
            df_path = os.path.join(config.COUNTRY_EMB,"{}.pkl".format(c))
            country_df = pd.read_pickle(df_path)
            country_df=country_df[country_df['features'].notnull()]
            
            # clean up df
            print("... transform bag of words features ...")
            country_df=pd.concat([country_df,#.drop(['features'],axis=1),
                                   country_df['features'].apply(pd.Series)],
                                    axis=1)
            # get embeding
            print("... get embeding for title and snip ...")
            country_df['snip_emb'] = country_df['features'].apply(lambda x: DP.get_emb_from_dict(x,'snip'))
            country_df['title_emb'] = country_df['features'].apply(lambda x: DP.get_emb_from_dict(x,'title'))

            # export to file 
            out_path = os.path.join(config.COUNTRY_EMB,"{}_emb.pkl".format(c))
            country_df.to_pickle(out_path)
    
    #%%
##%%
#    x = "I have no warn fear in my heart"
#    print(DP.process_from_json(country_df['data_path'].iloc[-2]))