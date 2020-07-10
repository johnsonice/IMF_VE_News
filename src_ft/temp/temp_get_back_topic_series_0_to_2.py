#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/20/2020

@author: apsurek
"""
## add country metadata
## data exploration, descripitive analysis

import sys,os
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
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

read_from = '/data/News_data_raw/FT_WD_research/doc_meta/a_part{}_doc_details_crisis_topic_ldaviz_t100.pkl'
write_to = '/data/News_data_raw/FT_WD_research/topiccing/series_savepoint_part{}.pkl'

for i in range(3):
    this_df_pkl = read_from.format(i)
    this_df = pd.read_pickle(this_df_pkl)
    topic_series_subset = this_df['ldaviz_t100_predicted_topics'][i*200000, (i+1)*200000]

    write_to_pkl = write_to.format(i)
    print("Wrote up to part {} at file {}".format(i, write_to_pkl))

