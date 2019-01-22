#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:42:19 2018

@author: chuang
"""
import os 
import sys
import pandas as pd
pd.set_option('display.max_columns',10)
#import warnings 
#%%
sys.path.insert(0,'./libs')
from crisis_points import crisis_points

## global arguments
MODE = 'test'# 'real'
SAMPLE_LIMIT= 500000            ## set max doc number, to fit into your memory 
COUNTRY_FREQ_PERIOD = 'month'   ## for country specific bow calculation 
WEIGHTED = False                 ## do we want to weighted average on similar words when doing evaluation
VERBOSE = True

## file specific inputs ##
countries=list(crisis_points.keys())
common_terms = ['he','him','she','her','that','if','me','about','over']

targets= ['fear','worry','concern','risk','threat','warn','maybe','may','possibly','could',
         'perhaps','uncertain','say','feel','predict','tell','believe','think','recession',
         'financial_crisis','crisis','depression','shock']
smooth_window_size = 24 # put as months , even if for quarterly data, put it as months
                        # it will automatically convert to quarterly
months_prior = 24        # same here, put as months
topn = 15
eval_end_date = {'q':'2001Q4',
                 'm':'2001-12'}  # or None


## Global folder path ##
RAW_DATA_PATH = '/data/News_data_raw/Financial_Times_processed'

PROCESSING_FOLDER = '/data/News_data_raw/FT_WD'
DOC_META = os.path.join(PROCESSING_FOLDER,'doc_meta')
JSON_LEMMA = os.path.join(PROCESSING_FOLDER,'json_lemma')

MODELS = os.path.join(PROCESSING_FOLDER,'models')
NGRAMS = os.path.join(MODELS,'ngrams')
VS_MODELS = os.path.join(MODELS,'vsms')

SEARCH_TERMS = os.path.join(PROCESSING_FOLDER,'search_terms')
BOW_TFIDF_DOCS = os.path.join(PROCESSING_FOLDER,'bow_tfidf_docs')
FREQUENCY = os.path.join(PROCESSING_FOLDER,'frequency')

EVAL = os.path.join(PROCESSING_FOLDER,'eval')
if WEIGHTED:
    EVAL = os.path.join(PROCESSING_FOLDER,'eval_weighted')

EVAL_WG = os.path.join(EVAL,'word_groups')
EVAL_TS = os.path.join(EVAL,'time_series')

## global file path ##
DOC_META_FILE = os.path.join(DOC_META,'doc_details_crisis.pkl')
AUG_DOC_META_FILE = os.path.join(DOC_META,'doc_details_crisis_aug.pkl')
PHRASER = os.path.join(NGRAMS,'2grams_default_10_20_NOSTOP')
W2V = os.path.join(VS_MODELS,'word_vecs_5_50_200')
EXPERT_TERMS = os.path.join(PROCESSING_FOLDER,'search_terms','expert_terms.csv')



#%%

def maybe_create(f):
    if os.path.exists(f):
        pass
    else:
        os.mkdir(f)
        print('New folder created: {}'.format(f))
        
        
if __name__ == "__main__":
    folders = [RAW_DATA_PATH,PROCESSING_FOLDER,SEARCH_TERMS,DOC_META,DOC_META,JSON_LEMMA,MODELS,NGRAMS,VS_MODELS,BOW_TFIDF_DOCS,
               FREQUENCY,EVAL,EVAL_WG,EVAL_TS]
    weights = [DOC_META_FILE,PHRASER,W2V]
    
    for f in folders:
        maybe_create(f)

    for w in weights:
        if not os.path.isfile(w):
            print('File not exist:{}'.format(w))

            
        
        
        
    
    
    
