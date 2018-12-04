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
sys.path.insert(0,'./libs')
import pandas as pd
from collections import defaultdict
from stream import DocStreamer_fast
from crisis_points import crisis_points
from region_mapping import region
import argparse
<<<<<<< HEAD
import config 

def list_period_docs(time_df, corp_path, period, countries = config.countries):
    # Gather lists of docs belonging to each period
    print("gathering doc lists...")
           
    total_docs = len(time_df)
    
=======

def list_period_docs(time_df, corp_path, period):
    # Gather lists of docs belonging to each period
    print("gathering doc lists...")
    period_dict = defaultdict(list)
    total_docs = len(time_df)
>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a
    for i, (id, data) in enumerate(time_df.iterrows()):
        print('\r\t{} of {} processed'.format(i, total_docs), end=' ')
        period_dict[data[period]].append(corp_path + "/" + id + ".json")
    return period_dict

<<<<<<< HEAD
def country_period_filter(time_df,country,period):
    
    time_df['filter_country'] = time_df['country'].apply(lambda c: country in c)
    df = time_df['data_path'][(time_df['filter_country'] == True)&(time_df['quarter'] == period)]
    
    return df.tolist()
    


def get_country_freqs(countries, period_choice, time_df, uniq_periods,outdir,phraser):
=======

def get_country_freqs(countries, corpus, period_choice, period_dict, outdir,phraser):
>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a

    # Get frequency data for each country supplied
    print("\nCounting Word Freqs...")
    for country in countries:
        print("\nWorking on {}".format(country))
        freqs = {}
<<<<<<< HEAD
        #for i, (period, doc_list) in enumerate(period_dict.items()):
        for i, period in enumerate(uniq_periods):
            print("\r\tworking on period {} of {}...".format(i, len(uniq_periods)), end=' ')
            p_freqs = defaultdict(int)
            doc_list = country_period_filter(time_df,country,period)
            doc_list = [os.path.join(config.JSON_LEMMA,os.path.basename(p)) for p in doc_list]
            print("Files to process: {}".format(len(doc_list)))
#            streamer = DocStreamer_fast(doc_list, language='en', regions=[region[country]], region_inclusive=True,
#                                    title_filter=[country],
#                                    phraser=phraser,lemmatize=False).multi_process_files(workers=int(os.cpu_count()/2),chunk_size = 500)
            streamer = DocStreamer_fast(doc_list, language='en',phraser=phraser,
                                        lemmatize=False)#.multi_process_files(workers=2,chunk_size = 100)
=======
        for i, (period, doc_list) in enumerate(period_dict.items()):
            print("\r\tworking on period {} of {}...".format(i, len(uniq_periods)), end=' ')
            p_freqs = defaultdict(int)
            streamer = DocStreamer_fast(doc_list, language='en', regions=[region[country]], region_inclusive=True,
                                    title_filter=[country],
                                    phraser=phraser,lemmatize=False).multi_process_files(workers=int(os.cpu_count()/2),chunk_size = 500)

>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a
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

        # Fill NAs as 0
        freqs_df = pd.DataFrame(freqs).fillna(0)
        # make sure columns are in ascending order
        freqs_df = freqs_df[freqs_df.columns.sort_values()] 

        # write pkl
<<<<<<< HEAD
        out_pkl = os.path.join(outdir, '{}_{}_word_freqs.pkl'.format(country, period_choice))
        freqs_df.to_pickle(out_pkl)

        # write csv
        out_csv = os.path.join(outdir, '{}_{}_word_freqs.csv'.format(country, period_choice))
        freqs_df.to_csv(out_csv)

class args_class(object):
    def __init__(self, corpus=config.JSON_LEMMA,doc_deets=config.AUG_DOC_META_FILE,
                 countries=crisis_points.keys(),period=config.COUNTRY_FREQ_PERIOD,out_dir=config.FREQUENCY,
                 phraser=config.PHRASER):
=======
        out_pkl = os.path.join(outdir, '{}_{}_{}_word_freqs.pkl'.format(country, os.path.basename(corpus), period_choice))
        freqs_df.to_pickle(out_pkl)

        # write csv
        out_csv = os.path.join(outdir, '{}_{}_{}_word_freqs.csv'.format(country, os.path.basename(corpus), period_choice))
        freqs_df.to_csv(out_csv)

class args_class(object):
    def __init__(self, corpus='../data/processed_json',doc_deets='../data/doc_meta/doc_details_full.pkl',
                 countries=crisis_points.keys(),period='quarter',out_dir='../data/frequency',
                 phraser='../models/ngrams/2grams_default_10_20_NOSTOP'):
>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a
        self.corpus = corpus
        self.doc_deets = doc_deets
        self.countries = countries
        self.period = period
        self.out_dir = out_dir
        self.phraser=phraser
        
<<<<<<< HEAD
#%%
=======

>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--countries', nargs='+', help='countries to get freq for',
                            default=crisis_points.keys())
        parser.add_argument('-corp', '--corpus', action='store', dest='corpus', 
                            default='../cleaned',required=True)
        parser.add_argument('-deets', '--doc_details', action='store', dest='doc_deets', 
                            default='.../data/doc_meta/doc_details_full.pkl',required=True)
        parser.add_argument('-p', '--period', action='store', dest='period', 
                            default='quarter',required=True)
        parser.add_argument('-s', '--save_dir', action='store', dest='out_dir', 
                            default='../data/frequency',required=True)
        parser.add_argument('-p', '--phraser', action='store', dest='phraser', 
                            default='../data/phrase_model/2grams_default_10_20_NOSTOP',required=True)
        args = parser.parse_args()
    except:
<<<<<<< HEAD
        args = args_class(corpus=config.JSON_LEMMA,doc_deets=config.AUG_DOC_META_FILE,
                          out_dir=config.FREQUENCY,period=config.COUNTRY_FREQ_PERIOD,
                          phraser=config.PHRASER,countries=config.countries)
=======
        args = args_class(corpus='../data/processed_json',doc_deets='../data/doc_meta/doc_details_crisis.pkl',out_dir='../data/frequency',period='quarter')

>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a

    # Data setup
    time_df = pd.read_pickle(args.doc_deets)
    uniq_periods = set(time_df[args.period])
<<<<<<< HEAD
    time_df = time_df[time_df['country_n']>0]      ## filter only docs with countries 
    #period_dict = list_period_docs(time_df, args.corpus, args.period)
#%%
    # obtain freqs
    print(args.period)
    get_country_freqs(args.countries, args.period, time_df, uniq_periods, args.out_dir,args.phraser)
=======
    period_dict = list_period_docs(time_df, args.corpus, args.period)

    # obtain freqs
    print(args.period)
    get_country_freqs(args.countries, args.corpus, args.period, period_dict, args.out_dir,args.phraser)
>>>>>>> 6c0fb2e5047bb403b4513e09123597e321d8e66a
