#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:11:34 2019

@author: chuang
"""

"""
frequency_eval.py

Description: Used to evaluate supplied terms and term groups wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval.py <TERM1> <TERM2> ...
NOTE: to see an explanation of optional arguments, use python3 frequency_eval.py --help
"""
import sys
import os
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
    
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))
import argparse
from gensim.models.keyedvectors import KeyedVectors
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
from frequency_utils import rolling_z_score,  signif_change #,aggregate_freq
import pandas as pd
import numpy as np
from mp_utils import Mp
import infer_config as config
import infer_utils
#%%
def read_grouped_search_words(file_path):
    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k,v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]
        temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
        search_groups[k]=temp_list
    return search_groups

def get_sim_words_set(args,word_group,vecs):
    assert isinstance(word_group,list)     
    sim_word_group = list()
    for w in word_group:
        words, weights = get_input_words_weights(args,
                                             w,
                                             vecs=vecs,
                                             weighted=args.weighted)
        sim_word_group.extend(words)
    sim_word_set = set(sim_word_group)
    return sim_word_set
    
def aggregate_freq(word_list,country, period='quarter', stemmed=False,frequency_path='../data/frequency', weights=None):
    assert isinstance(word_list, list), 'Must pass a list to aggregate_freq'
    s_flag = '_stemmed' if stemmed else ''
    if weights is None:
        weights = [1]*len(word_list)
    assert isinstance(weights, list), 'Must pass a list to aggregate_freq'
    assert len(weights)==len(word_list), 'Weights list must have the same length as word list'
    
    ww = zip(word_list,weights)

    #data_path = '/home/ubuntu/Documents/v_e/data/frequency/{}_cleaned_{}_word_freqs{}.pkl'.format(country, period, s_flag)
    data_path = os.path.join(frequency_path,'{}_{}_word_freqs{}.pkl'.format(country, period, s_flag))
    data = pd.read_pickle(data_path)
    ## fill nas only when document is missing
    cs = list(data.columns)
    for c in cs:
        if data[c].sum() == 0:
            pass
        else:
            data[c].fillna(0,inplace=True)
    
    freqs = [data.loc[word]*weight for word,weight in ww if word in data.index]
    grp_freq = sum(freqs)
    
    
    ##if none of the words are in corpus, frp_freq qill return 0 need to check befor proceed
    if isinstance(grp_freq,pd.Series):
        return grp_freq
    else:
        try:
            grp_freq = data.iloc[0]
        except:
            print(country)
            print('no data for the entire country')
            grp_freq = pd.Series(np.zeros(len(data.columns)),index=data.columns)
        grp_freq.values[:]=np.nan
    
    return grp_freq

def get_ts_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('-freq', '--frequency_path', action='store', dest='frequency_path', 
                        default=config.FREQUENCY)
    parser.add_argument('-ctm', '--current_ts_path', action='store', dest='current_ts_path', 
                        default=config.CURRENT_TS_PS)
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', 
                        default=config.W2V)    
    parser.add_argument('-tp', '--topn', action='store', dest='topn', type=int,
                        default=config.topn)
    parser.add_argument('-mp', '--months_prior', action='store', dest='months_prior', type=int,
                        default=config.months_prior)
    parser.add_argument('-win', '--window', action='store', dest='window', type=int,
                        default=config.smooth_window_size)
    parser.add_argument('-ct', '--countries', action='store', dest='countries',
                        default=config.countries)
    parser.add_argument('-pr', '--period', action='store', dest='period', 
                        default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-method', '--method', action='store', dest='method', 
                        default='zscore')
    parser.add_argument('-z', '--z_thresh', action='store', dest='z_thresh', type=int,
                        default=config.z_thresh)    
    parser.add_argument('-sims', '--sims', action='store', dest='sims', 
                        default=True)
    parser.add_argument('-weighted', '--weighted', action='store', dest='weighted', 
                        default=False)

    args = parser.parse_args()
    return args

    
class TS_generator(object):
    def __init__(self,args):
        self.args = args
        self.weights = None
        self.search_words_sets = self._get_search_words_sets()
        
        
        print('TS_generator created')
    
    def _get_search_words_sets(self):
        file_path = os.path.join(config.SEARCH_TERMS,config.GROUPED_SEARCH_FILE)  ## searh words name
        search_groups = read_grouped_search_words(file_path) 
        if self.args.sims:
            self.vecs = KeyedVectors.load(self.args.wv_path)
            
        if self.args.weighted:   
            raise Exception('for now, this module only works for unweighted calculation')
            print('Weighted flag = True ; Results are aggregated by weighted sum....')
        else:
            search_words_sets = dict()
            for k,v in search_groups.items():
                search_words_sets[k]=list(get_sim_words_set(self.args,search_groups[k],self.vecs)) ## turn set to list

        return search_words_sets
        
    def export_country_ts(self,country,export=True):
        series_wg = list()
        for k,words in self.search_words_sets.items(): 
            word_groups = words
            df = aggregate_freq(word_groups, country,period=self.args.period,stemmed=False,
                                frequency_path=self.args.frequency_path,
                                weights=None)

            df.name = k
            series_wg.append(df)

        df_all = pd.concat(series_wg,axis=1)
        #df_all.fillna(0,inplace=True)
        
        ####################################################
        ####################################################
        ## need to change latter, it is hard coded 
        ####################################################
        ####################################################
        
        if export:
            out_csv = os.path.join('/data/News_data_raw/Production/data/time_series_current_month/', 
                                   'agg_{}_{}_z{}_time_series.csv'.format(country,
                                                                            self.args.period,
                                                                            self.args.z_thresh))
            df_all.to_csv(out_csv)
        
        return country,df_all

    
#%%
        
if __name__ == '__main__':
        
    ts_args = ts_args(config)
    ts_generator = TS_generator(ts_args)
#    for country in config.countries:
#        print(country)
#        c,d = ts_generator.export_country_ts(country)
#    country = "uruguay"
#    c,d = export_country_ts(country)
#
    mp = Mp(config.countries,ts_generator.export_country_ts)
    res = mp.multi_process_files(chunk_size=1,workers=15)


