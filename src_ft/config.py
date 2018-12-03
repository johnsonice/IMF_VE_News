#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:42:19 2018

@author: chuang
"""
import os 
import sys
sys.path.insert(0,'./libs')
from crisis_points import crisis_points


MODE = 'test'# 'real'
SAMPLE_LIMIT= 2000000  ## set max doc number, to fit into your memory 
COUNTRY_FREQ_PERIOD = 'quarter'


## Global folder path ##
RAW_DATA_PATH = '/data/News_data_raw/Financial_Times_processed/FT_json_historical'

PROCESSING_FOLDER = '/data/News_data_raw/FT_WD'
DOC_META = os.path.join(PROCESSING_FOLDER,'doc_meta')
JSON_LEMMA = os.path.join(PROCESSING_FOLDER,'json_lemma')

MODELS = os.path.join(PROCESSING_FOLDER,'models')
NGRAMS = os.path.join(MODELS,'ngrams')
VS_MODELS = os.path.join(MODELS,'vsms')

BOW_TFIDF_DOCS = os.path.join(PROCESSING_FOLDER,'bow_tfidf_docs')
FREQUENCY = os.path.join(PROCESSING_FOLDER,'frequency')


## global file path ##
DOC_META_FILE = os.path.join(DOC_META,'doc_details_crisis.pkl')
PHRASER = os.path.join(NGRAMS,'2grams_default_10_20_NOSTOP')


## file specific inputs ##
countries=crisis_points.keys()
targets= ['fear', 'worry&risk','crisis','stress']

#%%

def maybe_create():
    
if __name__ == "__main__":
    folders = [RAW_DATA_PATH,PROCESSING_FOLDER,DOC_META,DOC_META,JSON_LEMMA,MODELS,NGRAMS,VS_MODELS,BOW_TFIDF_DOCS,FREQUENCY]
    weights = [DOC_META_FILE,PHRASER]
    
    for f in folders:
        
    
    
    