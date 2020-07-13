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
import config_topiccing as config
import pandas as pd
from stream import MetaStreamer_uberfast as MetaStreamer
from mp_utils import Mp

def get_period_topic_average(this_period_df):

    topics_list_series = this_period_df['{}_predicted_topics'.format(model_name)].to_list()
    topics_dict = {x: [0] for x in range(0, num_topics)}
    divisor = this_period_df.shape[0]
    for topics_list in topics_list_series:
        for tup in topics_list:
            topics_dict[tup[0]][0] += tup[1]/divisor

    this_period_index_val = this_period_df[period][0]
    period_topics_df = pd.DataFrame(data=topics_dict.values(), columns=topics_dict.keys(),
                                    index=[this_period_index_val])

    return period_topics_df


def generate_country_time_series(countries, period, df, setup_name):
    save_folder = os.path.join(topiccing_folder, 'time_series', setup_name) #TMP

    for country in countries:
        print("Working on country {}".format(country))
        country_save_file = os.path.join(save_folder, "{}_100_topic_time_series.csv".format(country))
        df['filter_country'] = df['country'].apply(lambda c: country in c)
        this_country_df = df[df['filter_country']]
        df = df.drop(columns=['filter_country'])
        this_country_df = this_country_df.drop(columns=['filter_country'])
        unique_periods = set(this_country_df[period])

        for i, this_period in enumerate(unique_periods):
            print("\r\tworking on period {} of {}...".format(i, len(unique_periods)), end=' ')
            this_period_df = this_country_df[this_country_df[period]==this_period]
            new_period_df = get_period_topic_average(this_period_df)
            if i == 0:
                topics_df = new_period_df
            else:
                topics_df = topics_df.join(new_period_df)
        topics_df.to_csv(country_save_file)
        print("Country time series save at {}".format(country_save_file))


if __name__ == '__main__':
    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_pkl = config.DOC_META_FILE
    json_data_path = config.JSON_LEMMA

    class_type_setups = config.class_type_setups
    model_name = "ldaviz_t100"
    topiccing_folder = "/data/News_data_raw/FT_WD_research/topiccing"

    series_saved_at = os.path.join(topiccing_folder, '{}_topic_meta'.format(model_name))
    series_base_file = os.path.join(series_saved_at, "series_savepoint_part{}.pkl")

    countries = config.countries
    num_topics = 100

    for setup in class_type_setups:
        setup_name = setup[0]
        meta_aug_pkl = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(setup_name))

        df = pd.read_pickle(meta_aug_pkl) # Read in the aug doc meta
        df = df[df['country_n'] > 0] # Drop not-identified country documents
        df['data_path'] = json_data_path+'/'+df.index + '.json'
        print('see one example : \n', df['data_path'].iloc[0])
        data_list = df['data_path'].tolist()
        data_length = len(data_list)
        period = config.COUNTRY_FREQ_PERIOD
        df = df.filter([period, 'country', 'country_n']) # Drop unnecessary columns from mem

        files_to_read = list(os.walk(series_saved_at))[0][2]
        for file_index in range(len(files_to_read)):
            this_pickle = os.path.join(series_saved_at, files_to_read[file_index])
            ds = pd.read_pickle(this_pickle)
            print("DS\n",ds.head())
            df = df.join(ds, how="left")
            print("DF\n",df.head())

            print("Read {} files, this one {}".format(file_index, this_pickle))

        generate_country_time_series(countries, period, df, setup_name)

