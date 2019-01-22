#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:29:53 2018

@author: chuang
"""

import os 
import pandas as pd
import sys
sys.path.insert(0,'./libs')
from stream import MetaStreamer_fast as MetaStreamer
import ujson as json
import config

#%%
from gensim.models.keyedvectors import KeyedVectors
vecs = KeyedVectors.load(config.W2V)
#%%
terms_dict = {}
for t in config.targets:
    print(t)
    try:
        words = vecs.wv.most_similar(t, topn=16)
        terms_dict[t] = [",".join([w[0] for w in words])]
        print(words)
    except:
        words = vecs.wv.most_similar(t.split("_"), topn=16)
        terms_dict[t] = [",".join([w[0] for w in words])]
        print(words)
    print('\n')
    
df = pd.DataFrame(terms_dict).T
#%%
df.to_csv(os.path.join(config.SEARCH_TERMS,'w2v_search_terms_results.csv'))
