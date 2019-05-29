#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:10:02 2019

@author: chuang
"""

import sys
import os
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
    
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))
#import argparse
from gensim.models.keyedvectors import KeyedVectors
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
from frequency_utils import rolling_z_score,  signif_change #,aggregate_freq
from country_freq_generator import Freq_generator 
from stream import FileStreamer_fast
from time_series_generator import TS_generator, get_ts_args
import pandas as pd
import numpy as np
from mp_utils import Mp
import infer_config as config
#import infer_utils
from collections import Counter
import re

#%%

class Tool_tips_generator(object):
    
    def __init__(self,config):
        self.config = config
        self.Fg = Freq_generator(config.DOC_META_FILE)
        self.periods = list(self.Fg.uniq_periods)
        self.search_words_sets = TS_generator(get_ts_args(config)).search_words_sets
        self.reg = re.compile(r'\bhttp.*\b')
    
        print('Tool_tips_generator created')
    
    def get_country_df(self,country,period,search_words_sets):
        country_docs = self.Fg.country_period_filter(self.Fg.full_time_df,country,period)
        
        if len(country_docs)==0:
            print('\n{} has no documents for period {}\n'.format(country,period))
            return []
        
        streamer = FileStreamer_fast(country_docs, language='en',phraser=self.config.PHRASER,
                                                    stopwords=[], lemmatize=False).multi_process_files(workers=2,chunk_size = 100)
        df_docs = pd.DataFrame(streamer)
        df_docs = self.produce_document_stats(df_docs,search_words_sets=search_words_sets)
        return df_docs
    
    def get_counts(self,token_list,search_sets):
        token_counter = Counter(token_list)
        res = 0 
        for i in search_sets:
            res += token_counter[i]
        return res
    
    def generate_tool_tip(self,df,sort_by,topn=3):
        res_df = df.sort_values(by=sort_by,ascending=False)[:topn]
        tool_top = ''
        for i,row in res_df.iterrows():
            tool_top += '\nTitle: {}\n{}\nkeywords count: {}\n'.format(row['title'],row['weburl'],row[sort_by])
            #format(row['title'],r['weburl']) 
            
        return tool_top
    
    def produce_document_stats(self,df,search_words_sets):
        for k,v in search_words_sets.items():
            df[k] = df['body'].apply(self.get_counts,args=(v,))
        
        return df 
    
    def load_historical_df(self,hist_df_path):
        if os.path.isfile(hist_df_path):
            old_df = pd.read_pickle(hist_df_path)
        else:
            return None
        return old_df
    
    @staticmethod
    def append_update_df(new_df,old_df):
        
        for p in new_df['time'].unique():
            if p in old_df['time'].unique():
                old_df = old_df[old_df['time']!=p]
                #print(old_df['time'].unique())
                old_df=old_df.append(new_df[new_df['time']==p],ignore_index=True)
            else:
                #print(old_df['time'].unique())
                old_df=old_df.append(new_df[new_df['time']==p],ignore_index=True)
                
        old_df.sort_values(by=['time','country_name','indexes'],inplace=True)
        #print(old_df['time'].unique())
        return old_df
    
    def _split_tooltip(self,tt):
        """split tooltips to titles and links"""
        links = self.reg.findall(tt)
        tt_s = self.reg.sub("",tt)
        tt_s = tt_s.replace("\n\n","\n").replace("/\n","")
        tt_s = [tt_s]
        tt_s.extend(links)
        return tt_s
    
    def split_tooltips(self,df,topn):
        """ update the extended columns everytime"""
        df = df[['time','country_name','indexes','tool_tips']]
        tts = df['tool_tips'].values.tolist()
        res = [self._split_tooltip(t) for t in tts]
        column_names = ['refined_tooltip']
        column_names.extend(['link{}'.format(i) for i in range(topn)])
        res_df = pd.DataFrame(res,columns=column_names)
        new = df.join(res_df)
        return new
    
    def get_tool_tips_df(self,topn=3):
        res_list = []
        for country in self.config.countries:
            for period in self.periods:
                #print(country)
                df_docs = self.get_country_df(country,period,self.search_words_sets)
                if len(df_docs)>0:
                    df_stats = self.produce_document_stats(df_docs,self.search_words_sets)
                    for k in self.search_words_sets.keys():
                        tt = self.generate_tool_tip(df_stats,k,topn)
                        res_list.append([period,country,k,tt])
        
        df_tt = pd.DataFrame(res_list,columns=['time','country_name','indexes','tool_tips'])
        old_df = self.load_historical_df(os.path.join(config.HISTORICAL_INPUT,'tool_tips_df.pkl'))
        if old_df is not None:
            df_tt = self.append_update_df(df_tt,old_df)
        
        df_tt = self.split_tooltips(df_tt,topn)
        return df_tt
    
#%%
if __name__ == '__main__':      
    TG = Tool_tips_generator(config)
    df = TG.get_tool_tips_df(topn=3)
    ## over write historical data

    df.to_pickle(os.path.join(config.HISTORICAL_INPUT,'tool_tips_df.pkl'))

#%%

#old = pd.read_pickle(os.path.join(config.HISTORICAL_INPUT,'tool_tips_df.pkl'))


    





