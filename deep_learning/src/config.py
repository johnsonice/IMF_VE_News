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
sys.path.insert(0,'../libs')

#import warnings 
#%%
sys.path.insert(0,'./libs')
from crisis_points import crisis_points,country_dict,ll_crisis_points

## global arguments
MODE = 'test'# 'real'
SAMPLE_LIMIT= 500000            ## set max doc number, to fit into your memory 
COUNTRY_FREQ_PERIOD = 'month'   ## for country specific bow calculation 
WEIGHTED = False                 ## do we want to weighted average on similar words when doing evaluation
SIM = True
VERBOSE = True
## crisis defination 
crisis_defs = 'kr' # or 'll' or 'kr'
##GROUPED_SEARCH_FILE = 'final_topic_words_final.csv'
GROUPED_SEARCH_FILE = 'grouped_search_words_final.csv'
#GROUPED_SEARCH_FILE = 'expert_terms_final.csv'

smooth_window_size = 24 # put as months , even if for quarterly data, put it as months
                        # it will automatically convert to quarterly
months_prior = 24       # same here, put as months
#months_prior = 12 
z_thresh = 2.1            # how many standard deviations away we think that is a spike 
topn = 15
eval_end_date = {'q':'2001Q4',
                 'm':'2001-12'}  # or None



########################
## Global folder path ##
########################
RAW_DATA_PATH = '/data/News_data_raw/Financial_Times_processed'

PROCESSING_FOLDER = '/data/News_data_raw/FT_WD'
DOC_META = os.path.join(PROCESSING_FOLDER,'doc_meta')
JSON_LEMMA = os.path.join(PROCESSING_FOLDER,'json_lemma')
JSON_LEMMA_SMALL = os.path.join(PROCESSING_FOLDER,'json_lemma_small')
COUNTRY_EMB = os.path.join(PROCESSING_FOLDER,'country_emb')
CRISIS_DATES = os.path.join(PROCESSING_FOLDER,'crisis_date')

MODELS = os.path.join(PROCESSING_FOLDER,'models')
NGRAMS = os.path.join(MODELS,'ngrams')
VS_MODELS = os.path.join(MODELS,'vsms')
TOPIC_MODELS = os.path.join(MODELS,'topics')
BERT_MODEL=os.path.join(MODELS,'pre_trained_english')
TRAINED_DEEP_MODEL=os.path.join(MODELS,'trained_deeplearning_model')

SEARCH_TERMS = os.path.join(PROCESSING_FOLDER,'search_terms')
BOW_TFIDF_DOCS = os.path.join(PROCESSING_FOLDER,'bow_tfidf_docs')
FREQUENCY = os.path.join(PROCESSING_FOLDER,'frequency','csv')

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


## file specific inputs ##
countries=list(country_dict.keys())
common_terms = ['he','him','she','her','that','if','me','about','over']


def load_search_words(folder,path):
    file_path = os.path.join(folder,path)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        search_groups = df.to_dict()
        words_list = list()
        for k,v in search_groups.items():
            temp_list = [i for i in list(v.values()) if not pd.isna(i)]
            #temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
            words_list.extend(temp_list)
        words_list = list(set(words_list))
    else:
        words_list = None
        print('file path does not exist:{}'.format(file_path))
    return words_list

targets = load_search_words(SEARCH_TERMS,GROUPED_SEARCH_FILE)

#targets= ['fear','worry','concern','afraid','trouble','uneasy','nervous','anxious',
#          'risk','threat','warn','hazard','contagious','impact','infect','terror','danger',
#          'maybe','may','possibly','could','perhaps','uncertain','doubt','unsure',
#          'say','feel','predict','tell','believe','think','suggest','decide','propose','advise','hint','clue','speak','announce',
#          'financial&recession','financial_crisis','depression','financial&shock','financial&slump','financial&slack','financial&fall']

#targets= ['fear','worry','concern',
#          'risk','threat','warn',
#          'maybe','may','possibly','could','perhaps','uncertain',
#         'say','feel','predict','tell','believe','think','recession',
#         'financial_crisis','crisis','depression','shock']

#targets= ['able', 'enable', 'grow', 'adequately', 'benign', 'buoyant', 'buoyancy', 'calm', 'comfortable', 'confidence', 'confident', 'effective', 'enhance', 'favorable', 'favourable', 'favourably', 'healthy', 'improve', 'improvement', 'mitigate', 'mitigation', 'positive', 'positively', 'profits', 'profitable', 'rally', 'rebound', 'recover', 'recovery', 'resilience', 'resilient', 'smooth', 'solid', 'sound', 'stabilise', 'stabilize', 'stable', 'success', 'successful', 'successfully']




#%%

def maybe_create(f):
    if os.path.exists(f):
        pass
    else:
        os.mkdir(f)
        print('New folder created: {}'.format(f))
    
if __name__ == "__main__":
    folders = [RAW_DATA_PATH,PROCESSING_FOLDER,SEARCH_TERMS,DOC_META,
               DOC_META,JSON_LEMMA,JSON_LEMMA_SMALL,MODELS,NGRAMS,TOPIC_MODELS,
               VS_MODELS,BOW_TFIDF_DOCS,
               FREQUENCY,EVAL,EVAL_WG,EVAL_TS]
    weights = [DOC_META_FILE,PHRASER,W2V]
    
    for f in folders:
        maybe_create(f)

    for w in weights:
        if not os.path.isfile(w):
            print('File not exist:{}'.format(w))

            
        
        
        
    
    
    
