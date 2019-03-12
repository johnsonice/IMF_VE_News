# -*- coding: utf-8 -*-

import pickle
import os 
import pandas as pd
import sys
from spacy.lang.en.stop_words import STOP_WORDS
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config

#%%
def clean_search_terms(search_terms):
    sts = [s.lower().strip('\xa0') for s in search_terms]
    return sts

def filter_search_terms(search_terms,vecs,STOP_WORDS=STOP_WORDS):
    vocabs = vecs.wv.vocab.keys()
    sts = []
    for t in search_terms:
        tempt = t.replace(" ","_")
        if tempt in vocabs:
            sts.append(tempt)
        else:
            temp_ct = []
            for tt in t.split():
                if tt in vocabs and tt not in STOP_WORDS:
                    temp_ct.append(tt)
                else:
                    continue
            if len(temp_ct) > 0: 
                sts.append("&".join(temp_ct))
    
    return sts
            

def read_grouped_search_words(file_path):
    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k,v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]
        temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
        search_groups[k]=temp_list
    return search_groups


#%%
from gensim.models.keyedvectors import KeyedVectors
vecs = KeyedVectors.load(config.W2V)

#%%
folder_path = os.path.join(config.PROCESSING_FOLDER,'search_terms','experts')
file_name1 = 'expert_terms_v1.csv'
file_name2 = 'expert_terms_v2.csv'
search_terms = config.load_search_words(folder_path,file_name1)
search_terms.extend(config.load_search_words(folder_path,file_name2))
search_terms = clean_search_terms(search_terms)
search_terms = list(set(search_terms))
print(search_terms[:10])
ss = filter_search_terms(search_terms,vecs)

#%%
df = pd.DataFrame(ss,columns=['expert_terms'])
df.to_csv(os.path.join(folder_path,'expert_terms_final.csv'),index=False)
#%%
sss = config.load_search_words(folder_path,'expert_terms_final.csv')
print(sss[:10])