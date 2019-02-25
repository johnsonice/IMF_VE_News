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
from gensim.models.keyedvectors import KeyedVectors
vecs = KeyedVectors.load(config.W2V)
#%%
terms = ['fear','worry','concern','risk','threat','warn','maybe','may','possibly','could',
         'perhaps','uncertain','say','feel','predict','tell','believe','think','recession',
         'financial_crisis','crisis','depression','shock']
terms = [['music'],['radio']]
for t in terms:
    print(t)
    try:
        print(vecs.wv.most_similar(t, topn=20))
    except:
        print(vecs.wv.most_similar(t.split("_"), topn=20))
    print('\n')
#%%
def read_grouped_search_words(file_path):
    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k,v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]
        temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
        search_groups[k]=temp_list
    return search_groups

file_path = os.path.join(config.SEARCH_TERMS,'model_100_search_terms.csv')
search_groups = read_grouped_search_words(file_path)  
#%%
#terms = ['financial']

for t in search_groups['1']:
    print(t)
    try:
        print(vecs.wv.most_similar(t, topn=20))
    except:
        print(vecs.wv.most_similar(t.split("_"), topn=20))
    print('\n')