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
from gensim.models.keyedvectors import KeyedVectors
import config

#%%
def get_list_if_in_vocab(vecs,words):
    vocabs = vecs.wv.vocab.keys()
    final_words = list()
    for w in words:
        if w in vocabs:
            final_words.append(w)
        else:
            print("{} : not in vocabulary".format(w))
    
    return final_words

def read_grouped_search_words(file_path):
    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k,v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]
        temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
        search_groups[k]=temp_list
    return search_groups
    

#%%
## load w2v
vecs = KeyedVectors.load(config.W2V)

## check if all words are in vocabulary 
file_path = os.path.join(config.SEARCH_TERMS,'grouped_search_words_final.csv')
search_groups = read_grouped_search_words(file_path)  
wordlist = [w for wg in search_groups['risk_language'] for w in wg]
res = get_list_if_in_vocab(vecs,wordlist)

#%%
terms_dict = {}
for t in config.targets:
    print(t)
    try:
        words = vecs.wv.most_similar(t, topn=15)
        terms_dict[t] = [",".join([w[0] for w in words])]
        print(words)
    except:
        try:
            words = vecs.wv.most_similar(t.split("_"), topn=15)
            terms_dict[t] = [",".join([w[0] for w in words])]
            print(words)
        except:
            words = vecs.wv.most_similar(t.split("&"), topn=15)
            terms_dict[t] = [",".join([w[0] for w in words])]
            print(words)
    print('\n')
    
df = pd.DataFrame(terms_dict).T
#%%
df.to_csv(os.path.join(config.SEARCH_TERMS,'w2v_search_terms_results.csv'))
