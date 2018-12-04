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
#terms = ['fear','worry','concern','risk','threat','warn','maybe','may','possibly','could',
#         'perhaps','uncertain','say','feel','predict','tell','believe','think','recession',
#         'financial_crisis','crisis','depression','shock']

terms = ['financial']
for t in terms:
    print(t)
    print(vecs.wv.most_similar(['financial','crisis'], topn=20))
    print('\n')