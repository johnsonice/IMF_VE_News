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
from stream import MetaStreamer_uberfast as MetaStreamer
from mp_utils import Mp


def country_period_filter(time_df, country, period):
    time_df['filter_country'] = time_df['country'].apply(lambda c: country in c)
    df = time_df['data_path'][(time_df['filter_country'] == True) & (time_df[args.period] == period)]

    return df.tolist()


def generate_country_time_series(countries, period, time_df, uniq_periods, out_dir, phraser, class_type):



if __name__ == '__main__':
    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_pkl = config.DOC_META_FILE
    json_data_path = config.JSON_LEMMA

    df = pd.read_pickle(meta_pkl)

    class_type_setups = config.class_type_setups
    model_name = "ldaviz_t100"
    topiccing_folder = "/data/News_data_raw/FT_WD_research/topiccing"

    series_saved_at = os.path.join(topiccing_folder, '{}_topic_meta'.format(model_name))
    series_base_file = os.path.join(series_saved_at, "series_savepoint_part{}.pkl")

    df['data_path'] = json_data_path+'/'+df.index + '.json'
    print('see one example : \n', df['data_path'].iloc[0])
    pre_chunked = True  # The memory will explode otherwise
    data_list = df['data_path'].tolist()
    del df
    data_length = len(data_list)

    part_i = 0
    partition_start = 0
    partition_size = 200000
    class_type_setups = config.class_type_setups

    if len(sys.argv) > 1:
        part_i = int(sys.argv[1])
        partition_start = partition_size * part_i

    while partition_start < data_length:
        partition_end = min(partition_start + partition_size, data_length)
        partition_save_file = os.path.join(topiccing_folder, "series_savepoint_part{}.pkl".format(part_i))

        series_file = series_base_file.format(part_i)
        topic_series = pd.read_pickle(series_file)

        pre_chunk_size = 10000
        index = []
        predicted_topics = []
        chunky_index = partition_start
        while chunky_index < partition_end:
            chunk_end = min(chunky_index + pre_chunk_size, partition_end)
            streamer = MetaStreamer(data_list[chunky_index:chunk_end])

            news = streamer.multi_process_files(workers=10, chunk_size=500)
            del streamer  # free memory

            mp = Mp(news, generate_time_series)  # TMP
            # mp = Mp(news, get_countries_by_count_2)
            del news

            # Discard docs without any countries

            # Transform doc_meta time column to the period_start - based on "config.COUNTRY_FREQ_PERIOD - a string"


            generate_country_time_series(countries,

            for setup in class_type_setups:
                setup_name = setup[0]
                meta_aug_pkl = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(setup_name))






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

