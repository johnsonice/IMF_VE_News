#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
frequency_country_specific_freqs.py

Description: retrieve and save country-specific word frequencies for each supplied country. The word 
freq data for each country will only be based on articles which either mention the country name in the
title or abstract, or which are labeled with the region code corresponding to that particular country. 

usage: python3 frequency_country_specific_freqs.py
NOTE: can be done for as many countries at a time as you want.
"""

import os
import sys
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
    
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))

import pandas as pd
from collections import defaultdict
from stream import DocStreamer_fast
from crisis_points import country_dict
import argparse
import infer_config as config 
import infer_utils
#%%

class Freq_generator():
    
    def __init__(self,doc_meta,period = config.COUNTRY_FREQ_PERIOD):
        self.full_time_df = pd.read_pickle(doc_meta)
        self.uniq_periods = set(self.full_time_df[period])
        self.period_freq = period
        ## get only docs with country labels 
        self.time_df = self.full_time_df[self.full_time_df['country_n']>0]
        print('Frequency generator initialized...')
        
    
    def country_period_filter(self,time_df,country,period):
        
        time_df['filter_country'] = time_df['country'].apply(lambda c: country in c)
        df = time_df['data_path'][(time_df['filter_country'] == True)&(time_df[self.period_freq] == period)]
        
        return df.tolist()
    
    def get_country_freqs(self,countries, corpus, period_choice,outdir,phraser):
        infer_utils.maybe_create(outdir)
    
        # Get frequency data for each country supplied
        print("\nCounting Word Freqs...")
        for country in countries:
            print("\nWorking on {}".format(country))
            freqs = {}
            #for i, (period, doc_list) in enumerate(period_dict.items()):
            for i, period in enumerate(self.uniq_periods):
                print("\r\tworking on period {} of {}...".format(i, len(self.uniq_periods)), end=' ')
                p_freqs = defaultdict(int)
                doc_list = self.country_period_filter(self.time_df,country,period)
                doc_list = [os.path.join(corpus,os.path.basename(p)) for p in doc_list]
                print("Files to process: {}".format(len(doc_list)))
    #            streamer = DocStreamer_fast(doc_list, language='en', regions=[region[country]], region_inclusive=True,
    #                                    title_filter=[country],
    #                                    phraser=phraser,lemmatize=False).multi_process_files(workers=int(os.cpu_count()/2),chunk_size = 500)
                streamer = DocStreamer_fast(doc_list, language='en',phraser=phraser,
                                            stopwords=[], lemmatize=False)#.multi_process_files(workers=2,chunk_size = 100)
                # count
                for doc in streamer:
                    for token in doc:
                        p_freqs[token] += 1
    
                # Normalize
                total_tokens = sum(p_freqs.values())
                per_thousand = total_tokens / 1000
                p_freqs = {k: v / per_thousand for k, v in p_freqs.items()}
    
                # Add to master dict
                freqs[period] = p_freqs
    
            # Fill NAs as 0 or not
            freqs_df = pd.DataFrame(freqs)#.fillna(0)
            # make sure columns are in ascending order
            freqs_df = freqs_df[freqs_df.columns.sort_values()] 
    
            # write pkl
            out_pkl = os.path.join(outdir, '{}_{}_word_freqs.pkl'.format(country, period_choice))
            freqs_df.to_pickle(out_pkl)
    
            # write csv
            out_csv = os.path.join(outdir, '{}_{}_word_freqs.csv'.format(country, period_choice))
            freqs_df.to_csv(out_csv)

        
#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--countries', nargs='+', help='countries to get freq for',
                        default=country_dict.keys())
    parser.add_argument('-corp', '--corpus', action='store', dest='corpus', 
                        default='/data/News_data_raw/Production/data/input_processed_current_month/')
    parser.add_argument('-meta', '--doc_details', action='store', dest='doc_meta', 
                        default='/data/News_data_raw/Production/data/meta/doc_details_crisis.pkl')
    parser.add_argument('-pe', '--period', action='store', dest='period', default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-s', '--save_dir', action='store', dest='out_dir',
                        default='/data/News_data_raw/Production/data/frequency_current_month/')
    parser.add_argument('-pa', '--phraser', action='store', dest='phraser', default=config.PHRASER)
    args = parser.parse_args()

#%%
    # obtain freqs
    print(args.period)
    Fg = Freq_generator(args.doc_meta)
    Fg.get_country_freqs(args.countries, args.corpus,args.period, args.out_dir,args.phraser)
