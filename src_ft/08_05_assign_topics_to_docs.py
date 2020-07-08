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
model_address = os.path.join(model_folder, this_model)
loaded_model = gensim.models.ldamodel.LdaModel.load(model_address)


def get_topic_prediction(text):
    tokens = text.split()
    bowed = common_dictionary.doc2bow(tokens)
    text_topics = loaded_model.get_document_topics(bowed, minimum_probability=0)

    #print(text_topics)
    #print(type(text_topics))

    return text_topics


def topic_this_document(article):
    '''
    Identifies the predicted topics for the specific article
    '''

    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    body = article['body'].lower() if article['body'] else None
    if body and title:
        title = "{} {}".format(title, body)
        topics = list(get_topic_prediction(title))
    elif title and snip:
        title = "{} {}".format(title, snip)
        topics = list(get_topic_prediction(title))
    elif body:
        topics = list(get_topic_prediction(body))
    elif title:
        topics = list(get_topic_prediction(title))
    elif snip:
        topics = list(get_topic_prediction(snip))
    else:
        topics = list()

    #print(article['an'], topics)

    return article['an'], topics


if __name__ == '__main__':
    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_pkl = config.DOC_META_FILE
    json_data_path = config.JSON_LEMMA

    df = pd.read_pickle(meta_pkl)

    class_type_setups = config.class_type_setups
    model_name = "ldaviz_t100"
    temp_pkl_file = "/data/News_data_raw/FT_WD_research/test/topic_data_series_t2.pkl"

    df['data_path'] = json_data_path+'/'+df.index + '.json'
    print('see one example : \n', df['data_path'].iloc[0])
    pre_chunked = True  # The memory will explode otherwise

    # Go through the files in chunks
    if pre_chunked:

        data_list = df['data_path'].tolist()
        del df
        pre_chunk_size = 10000
        chunky_index = 0
        data_length = len(data_list)
        index = []
        predicted_topics = []
        while chunky_index < data_length:
            if chunky_index%100000 == 0:
                print("Passed ", chunky_index, " files")
            chunk_end = min(chunky_index+pre_chunk_size, data_length)

            # streamer = MetaStreamer(data_list[chunky_index:chunk_end])
            streamer = MetaStreamer_SLOW(data_list[chunky_index:chunk_end])  # TMP

            news = streamer.multi_process_files(workers=10, chunk_size=1000)
            del streamer  # free memory

            mp = Mp(news, topic_this_document)  # TMP
            # mp = Mp(news, get_countries_by_count_2)

            topic_meta = mp.multi_process_files(workers=10, chunk_size=1000)

            index = [i[0] for i in topic_meta]
            country_list = [i[1] for i in topic_meta]

            if chunky_index != 0:
                read_series = pd.read_pickle(temp_pkl_file)
                add_series = pd.Series(country_list, name='{}_predicted_topics'.format(model_name),
                                       index=index)
                sum_series = read_series.append(add_series)
                del read_series
                del add_series
            else:
                sum_series = pd.Series(country_list, name='{}_predicted_topics'.format(model_name),
                                       index=index)

            del index
            del country_list

            #print("SUM SERIES:")
            #print(sum_series.head())

            sum_series.to_pickle(temp_pkl_file)
            del sum_series
            print("Wrote up to", chunky_index)

            chunky_index = chunk_end

            del topic_meta   # clear memory
            del mp  # clear memory

        ds = pd.read_pickle(temp_pkl_file)
        # os.remove("temp_in_processing.pkl") # put into final

        meta_root = config.DOC_META
        meta_aug = config.AUG_DOC_META
        meta_aug_pkl = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format('Min1'))
        meta_pkl = config.DOC_META_FILE

        df = pd.read_pickle(meta_pkl)  # Re-load deleted df - not multiplied when multiprocessing anymore
        new_df = df.join(ds)  # merge country meta
        new_df_file = os.path.join(meta_aug, 'a_doc_details_{}_topic_{}.pkl'.format('crisis', model_name))
        #new_df_file = "/data/News_data_raw/FT_WD_research/test/topic_test1.pkl"
        new_df.to_pickle(new_df_file)
        print('Topic document meta data saved at {}'.format(new_df_file))

        aug_df = pd.read_pickle(meta_aug_pkl)
        new_aug_df = aug_df.join(ds)
        new_aug_file = os.path.join(meta_aug, 'a_doc_details_{}_aug_{}_topic_{}.pkl'.format('crisis', 'Min1', model_name))
        #new_aug_file = "/data/News_data_raw/FT_WD_research/test/topic_aug_test1.pkl"
        new_aug_df.to_pickle(new_aug_file)
        print('Aug topic document meta data saved at {}'.format(new_aug_file))
