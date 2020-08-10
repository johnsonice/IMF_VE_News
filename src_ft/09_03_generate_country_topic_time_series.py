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
import pickle as pkl


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

    partition_size = 200000

    for setup in class_type_setups:
        setup_name = setup[0]
        meta_aug_pkl = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(setup_name))

        df = pd.read_pickle(meta_aug_pkl) # Read in the aug doc meta
        df['data_path'] = json_data_path+'/'+df.index + '.json'
        print('see one example : \n', df['data_path'].iloc[0])
        data_list = df['data_path'].tolist()
        data_length = len(data_list)
        period = config.COUNTRY_FREQ_PERIOD
        df = df.filter([period, 'country', 'country_n']) # Drop unnecessary columns from mem

        save_folder = os.path.join(topiccing_folder, 'time_series', setup_name)

        num_of_series = len(list(os.walk(series_saved_at))[0][2])

        for part_i in range(num_of_series):
            print("Working on part {}".format(part_i))
            partition_start = part_i*partition_size
            partition_end = min(partition_start + partition_size, data_length)

            part_df = df[partition_start:partition_end]

            this_series_file = series_base_file.format(part_i)
            ds = pd.read_pickle(this_series_file)
            part_df = part_df.join(ds, how="left")
            del ds

            part_df = part_df[part_df['country_n'] > 0]  # Drop 0-country documents

            for country in countries:
                print("Working on country {}".format(country))

                part_df['filter_country'] = part_df['country'].apply(lambda c: country in c)
                this_country_df = part_df[part_df['filter_country']]  # look at only this country
                part_df = part_df.drop(columns=['filter_country'])
                this_country_df = this_country_df.drop(columns=['filter_country'])
                unique_periods = set(this_country_df[period])

                # Do not want to create pickles for things with no observations
                if len(unique_periods) == 0:
                    break

                country_temp_pkl = os.path.join(save_folder, "{}_temp_in_process.pkl".format(country))

                # If pkl exits, this is not part0, and there needs to be appends
                if os.path.exists(country_temp_pkl):
                    temp_pkl_file = open(country_temp_pkl, 'rb')
                    all_periods_dict = pkl.load(temp_pkl_file)
                    temp_pkl_file.close()
                    print("Loaded pickle file {}".format(country_temp_pkl))
                else:
                    all_periods_dict = {}

                print('Processing {} periods'.format(len(unique_periods)))

                for i, this_period in enumerate(unique_periods):

                    this_period_df = this_country_df[this_country_df[period] == this_period]  # Look at only this period
                    topics_list_series = this_period_df['{}_predicted_topics'.format(model_name)].to_list()
                    num_new_docs = this_period_df.shape[0]  # Needed for divisor and further averages

                    if this_period in all_periods_dict:
                        topics_dict = all_periods_dict[this_period]
                        num_old_docs = topics_dict['num_docs']
                        num_docs = num_new_docs + num_old_docs
                        # Account for proportional weight of previously recorded observations
                        for topic_num in range(num_topics):
                            topics_dict[topic_num] *= num_old_docs/num_docs

                    else:
                        topics_dict = {x: 0 for x in range(num_topics)}
                        num_docs = num_new_docs

                    topics_dict.update({'num_docs': num_docs})
                    for topics_list in topics_list_series:

                        # Check for out-of-order-reading
                        if type(topics_list) is not list:
                            raise ValueError("READING IN WRONG ORDER in period {}".format(this_period))

                        for tup in topics_list:
                            topics_dict[tup[0]] += tup[1] / num_docs

                    all_periods_dict.update({this_period: topics_dict})
                    del topics_dict
                temp_pkl_file = open(country_temp_pkl, 'wb')
                pkl.dump(all_periods_dict, temp_pkl_file) # Kept for safety
                temp_pkl_file.close()

                if (part_i+1)*partition_size >= data_length: # Indicates last partition - test this TODO
                    if len(all_periods_dict) == 0:
                        print("Country {} had no observations".format(country))
                        break

                    index_list = list(all_periods_dict.keys())
                    columns = list(all_periods_dict[index_list[0]].keys())
                    data = [list(all_periods_dict[ind].values()) for ind in index_list]
                    this_country_df = pd.DataFrame(index=index_list, columns=columns, data=data)
                    country_save_file = os.path.join(save_folder, "{}_100_topic_time_series.csv".format(country))
                    this_country_df = this_country_df.sort_index()
                    this_country_df.to_csv(country_save_file)
                    print("Country time series save at {}".format(country_save_file))
                    del this_country_df

                del all_periods_dict

            del part_df
