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

this_model = "lda_model_tfidf_100_None_4"

if __name__ == '__main__':
    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_pkl = config.DOC_META_FILE

    model_name = "ldaviz_t100"
    temp_pkl_file = "/data/News_data_raw/FT_WD_research/test/topic_data_series_t4.pkl"
    topiccing_folder = "/data/News_data_raw/FT_WD_research/topiccing"

    files_to_read = [x[2] for x in os.walk(topiccing_folder)]
    for file_index in range(len(files_to_read)):
        this_pickle = os.path.join(topiccing_folder, files_to_read[file_index])
        if file_index == 0:
            ds = pd.read_pickle(this_pickle)
        else:
            ds = ds.join(pd.read_pickle(this_pickle))
        print("Read up to part {}".format(file_index))

    df = pd.read_pickle(meta_pkl)
    new_df = df.join(ds)  # merge country meta
    #new_df_file = os.path.join(meta_aug, 'doc_details_{}_topic_{}.pkl'.format('crisis', model_name))
    new_df_file = "/data/News_data_raw/FT_WD_research/test/topic_docu_test.pkl"
    new_df.to_pickle(new_df_file)
    print('Topic document meta data saved at {}'.format(new_df_file))
