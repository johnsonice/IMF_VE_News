#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:24:04 2018

@author: chuang
"""
import pandas as pd
import os 
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config 


def aggregate_freq(word_list,country, weights=None,period='quarter', stemmed=False,frequency_path='../data/frequency'):
    assert isinstance(word_list, list), 'Must pass a list to aggregate_freq'
    s_flag = '_stemmed' if stemmed else ''
    if weights is None:
        weights = [1]*len(word_list)
    assert isinstance(weights, list), 'Must pass a list to aggregate_freq'
    assert len(weights)==len(word_list), 'Weights list must have the same length as word list'
    
    ww = zip(word_list,weights)
    print(ww)
    #data_path = '/home/ubuntu/Documents/v_e/data/frequency/{}_cleaned_{}_word_freqs{}.pkl'.format(country, period, s_flag)
    data_path = os.path.join(frequency_path,'{}_{}_word_freqs{}.pkl'.format(country, period, s_flag))
    data = pd.read_pickle(data_path)
    freqs = [data.loc[word]*weight for word,weight in ww if word in data.index]
    grp_freq = sum(freqs)
    
    return grp_freq


#%%
    
word_list = ['i','you']
weights = [100,10]
country = 'argentina'
df = aggregate_freq(word_list,country,weights = weights,frequency_path = config.FREQUENCY)
print(df.head())