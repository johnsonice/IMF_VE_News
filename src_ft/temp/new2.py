#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 08:46:26 2018

@author: chuang
"""

from toolz import partition_all
from pathlib import Path
from joblib import Parallel, delayed
import thinc.extra.datasets
import plac
import spacy


#%%


model = 'en_core_web_lg'
n_jobs=4
batch_size=1000,
limit=10000


nlp = spacy.load("en_core_web_lg",disable=['tagger','ner','parser','textcat'])
print("Loaded model '%s'" % model)
test = 'i am a cat. They are cats. interesting.'
x = nlp(test)
for i in x:
    print(i.lemma_)
#%%
# load and pre-process the IMBD dataset
print("Loading IMDB data...")
data, _ = thinc.extra.datasets.imdb()
texts, _ = zip(*data[-limit:])
#%%
print("Processing texts...")
partitions = partition_all(batch_size, texts)
executor = Parallel(n_jobs=n_jobs)