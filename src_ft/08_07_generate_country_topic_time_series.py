#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/20/2020

@author: apsurek
"""
## add country metadata 
## data exploration, descripitive analysis 

import sys,os
sys.path.insert(0,'./libs')
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from crisis_points import country_dict
from nltk.tokenize import word_tokenize
from stream import MetaStreamer_fast as MetaStreamer
from stream import MetaStreamer_slow as MetaStreamer_SLOW
#import time 
from mp_utils import Mp
import re
import logging
import gensim
#plt.rcParams['figure.figsize']=(10,5)

f_handler = logging.FileHandler('err_log_7_1908.log')
f_handler.setLevel(logging.WARNING)

# TODO make flexible
corpus_path = os.path.join(config.BOW_TFIDF_DOCS, 'tfidf.mm')
corpus = gensim.corpora.MmCorpus(corpus_path)

common_dictionary_path = os.path.join(config.BOW_TFIDF_DOCS, 'dictionary')
common_dictionary = gensim.corpora.Dictionary.load(common_dictionary_path)

model_folder = "/data/News_data_raw/FT_WD/models/topics"
this_model = "lda_model_tfidf_100_None_4"
aug_file_to_read = os.path.join(config.AUG_DOC_META

if __name__ == '__main__':
    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_pkl = config.DOC_META_FILE
    json_data_path = config.JSON_LEMMA

    df = pd.read_pickle(meta_pkl)

    class_type_setups = config.class_type_setups
    model_name = "ldaviz_t100"
    temp_pkl_file = "/data/News_data_raw/FT_WD_research/test/topic_data_series_t4.pkl"
    topiccing_folder = "/data/News_data_raw/FT_WD_research/topiccing"

    df['data_path'] = json_data_path+'/'+df.index + '.json'
    print('see one example : \n', df['data_path'].iloc[0])
    pre_chunked = True  # The memory will explode otherwise
    data_list = df['data_path'].tolist()
    del df
    data_length = len(data_list)

    part_i = 0
    partition_start = 0
    partition_size = 200000

    if len(sys.argv) > 1:
        part_i = int(sys.argv[1])
        partition_start = partition_size * part_i

    # Sum the series together

    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_aug_pkl = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format('Min1'))
    meta_pkl = config.DOC_META_FILE

    ds = None ## TEMP
    df = pd.read_pickle(meta_pkl)  # Re-load deleted df - not multiplied when multiprocessing anymore
    new_df = df.join(ds)  # merge country meta
    new_df_file = os.path.join(meta_aug, 'a_part{}_doc_details_{}_topic_{}.pkl'.format(part_i,'crisis', model_name))
    #new_df_file = "/data/News_data_raw/FT_WD_research/test/topic_test1.pkl"
    new_df.to_pickle(new_df_file)
    print('Topic document meta data saved at {}'.format(new_df_file))

    aug_df = pd.read_pickle(meta_aug_pkl)
    new_aug_df = aug_df.join(ds)
    new_aug_file = os.path.join(meta_aug, 'a_part{}_doc_details_{}_aug_{}_topic_{}.pkl'.format(part_i,'crisis', 'Min1', model_name))
    #new_aug_file = "/data/News_data_raw/FT_WD_research/test/topic_aug_test1.pkl"
    new_aug_df.to_pickle(new_aug_file)
    print('Aug topic document meta data saved at {}'.format(new_aug_file))

