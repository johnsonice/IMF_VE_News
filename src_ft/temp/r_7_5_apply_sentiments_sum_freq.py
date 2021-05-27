"""
frequency_country_specific_freqs.py

Description: retrieve and save country-specific word frequencies for each supplied country. The word 
freq data for each country will only be based on articles which either mention the country name in the
title or abstract, or which are labeled with the region code corresponding to that particular country. 

usage: python3 frequency_country_specific_freqs.py
NOTE: can be done for as many countries at a time as you want.
"""
import logging
import os
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import pandas as pd
from stream import DocStreamer_fast
from crisis_points import country_dict
import crisis_points
#from region_mapping import region
import argparse
import config
from numpy.random import choice as np_choice

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# create a file handler
handler = logging.FileHandler('converting.log')
handler.setLevel(logging.WARNING)
# create logging format 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(handler)

def data_setup(doc_deets, period):
    time_df = pd.read_pickle(doc_deets)
    uniq_periods = set(time_df[period])
    time_df = time_df[time_df['country_n'] > 0]  ## filter only docs with countries

    return time_df, uniq_periods

def country_period_filter(time_df,country,period):
    
    time_df['filter_country'] = time_df['country'].apply(lambda c: country in c)
    df = time_df['data_path'][(time_df['filter_country'] == True)&(time_df[args.period] == period)]
    
    return df.tolist()

# Base - Words Counting
import numpy as np

def sum_words_freq(sentence, words):
    try:
        to_re = np.sum([sentence.count(x) for x in words])/len(sentence.split())
    except:
        print('SENTENCE ', sentence, '\nWORDS', words)
        to_re = np.NaN
    return to_re

def get_sentiments(doc, word_defs, ind):
    sent_df = pd.DataFrame(index=[ind])
    sentence = ' '.join(doc)
    divisor = len(doc)

    # Word counts / sum words
    for def_name in word_defs.columns:
        sent_df[def_name] = sum_words_freq(sentence, word_defs[def_name].dropna())/divisor
        if np.isnan(sent_df[def_name].values[0]):
            return None

    return sent_df

def get_country_freqs_sample(countries, period_choice, time_df, uniq_periods, outdir, phraser, filter_dict=None,
                             sample_size=25, word_defs=None):

    # Get frequency data for each country supplied
    print("\nCounting Word Freqs...")

    # Save docname, country, month, sentiments, and word hitrate (count/sum words) according to different definitions
    huge_doc_map = pd.DataFrame()

    small_doc_map = None
    for country in countries:
        print("\nWorking on {}".format(country))
        # for i, (period, doc_list) in enumerate(period_dict.items()):

        total_doc = 0
        n_outs = 1
        write_outs = n_outs = 1

        small_doc_map = None
        uniq_periods = np.array(sorted(list(uniq_periods)))


        for i, period in enumerate(uniq_periods):

            #print("\r\tworking on period {} of {}...".format(i, len(uniq_periods)), end=' ')

            doc_list_a = country_period_filter(time_df, country, period)

            #if len(doc_list_a) > sample_size:
            #    doc_list_a = np_choice(doc_list_a, size=sample_size)

            doc_list = [os.path.join(config.JSON_LEMMA, os.path.basename(p)) for p in doc_list_a]

            streamer = DocStreamer_fast(doc_list, language='en', phraser=phraser,
                                        stopwords=[], lemmatize=False).multi_process_files(workers=5, chunk_size=50)
            # count
            small_doc_map = pd.DataFrame()

            docnum = -1
            for doc in streamer:
                docnum += 1
                if doc is None:
                    continue

                tiny_doc_map = pd.DataFrame(index=[docnum],data={'doc_name': doc_list[docnum]})
                sentiments = get_sentiments(doc, word_defs, docnum) # Returns DataFrame
                if sentiments is None:
                    small_doc_map = None

                tiny_doc_map = pd.merge(tiny_doc_map, sentiments, left_index=True, right_index=True, how='outer')
                small_doc_map = small_doc_map.append(tiny_doc_map, ignore_index=True)
                #print('print 1\n\n',small_doc_map)

            small_doc_map['month'] = period
            small_doc_map['country'] = country
            small_doc_map['num_doc'] = docnum

            if small_doc_map is None:
                continue

            total_doc += docnum


            huge_doc_map = huge_doc_map.append(small_doc_map, ignore_index=True)
            if total_doc > 5000*n_outs:
                outname = os.path.join(outdir, '{}_doc_sentiment_map_{}.csv'.format(country, write_outs))
                huge_doc_map.to_csv(outname)
                huge_doc_map = pd.DataFrame()
                small_doc_map = pd.DataFrame()
                print('Saved up to {} inclusive with ending {}'.format(period, write_outs))
                n_outs += 1
                write_outs += 1

        if total_doc < 10000:
            outname = os.path.join(outdir, '{}_doc_sentiment_map.csv'.format(country))
        else:
            outname = os.path.join(outdir, '{}_doc_sentiment_map_{}.csv'.format(country,write_outs))
        print('Done {}'.format(country))

        #outname = os.path.join(outdir, 'doc_sentiment_map_test.csv')

        huge_doc_map.to_csv(outname)
        huge_doc_map = pd.DataFrame()

    return None

class args_class(object):
    def __init__(self, corpus=config.JSON_LEMMA,doc_deets=config.AUG_DOC_META_FILE,
                 countries=country_dict.keys(),period=config.COUNTRY_FREQ_PERIOD,out_dir=config.FREQUENCY,
                 phraser=config.PHRASER):
        self.corpus = corpus
        self.doc_deets = doc_deets
        self.countries = countries
        self.period = period
        self.out_dir = out_dir
        self.phraser=phraser
        
#%%
if __name__ == '__main__':

    class_type_setups = config.class_type_setups

    # Configure arguments (read from bash, or use default)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--countries', nargs='+', help='countries to get freq for',
                        default=config.countries)
    parser.add_argument('-corp', '--corpus', action='store', dest='corpus',
                        default=config.JSON_LEMMA)
    parser.add_argument('-deets', '--doc_details', action='store', dest='doc_deets',
                        default=config.AUG_DOC_META_FILE)
    parser.add_argument('-p', '--period', action='store', dest='period',
                        default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-s', '--save_dir', action='store', dest='out_dir',
                        default=config.FREQUENCY)
    parser.add_argument('-ph', '--phraser', action='store', dest='phraser',
                        default=config.PHRASER)
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', default=config.W2V)
    parser.add_argument('-f', '--wv_filter', action='store', dest='wv_filter', default='TRUE')
    args = parser.parse_args()

    word_defs_f = '/home/apsurek/IMF_VE_News/research/w2v_compare/all_sims_maps.csv'
    word_defs = pd.read_csv(word_defs_f).drop(columns='Unnamed: 0')
    word_defs = word_defs.drop(columns=['w2v_refined_0_pos', 'w2v_refined_0_neg','w2v_refined_1_pos',
                                        'w2v_refined_1_neg'])

    #class_type = 'Min1_AllCountry'
    #doc_deetz = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(class_type))
    doc_deetz = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_Min1_Thin.pkl')

    #args.out_dir = config.EVAL_WordDefs
    #args.out_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_new')
    args.out_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_new_test')

    sentiment_progress = pd.read_csv(os.path.join(config.AUG_DOC_META, 'sentiment_progress.csv'))
    possible_countries = sentiment_progress['aug_doc_countries'].values

    # Add all possible countries, from IMF defs and all others
    countries_to_sent = set()

    # KnR
    crisis_dict = crisis_points.crisis_points_TEMP_KnR
    countries_to_sent.update(set(crisis_dict.keys()))

    # LL
    crisis_dict = crisis_points.ll_crisis_points
    countries_to_sent.update(set(crisis_dict.keys()))

    # IMF all events
    crisis_dict = crisis_points.imf_gap_6_events
    countries_to_sent.update(set(crisis_dict.keys()))

    crisis_dict = crisis_points.imf_all_events
    countries_to_sent.update(set(crisis_dict.keys()))

    # Romer Romer
    crisis_dict = crisis_points.crisis_points_RomerNRomer
    countries_to_sent.update(set(crisis_dict.keys()))

    # LoDuca
    crisis_dict = crisis_points.crisis_points_LoDuca
    countries_to_sent.update(set(crisis_dict.keys()))

    # Reinhart Rogoff
    crisis_dict = crisis_points.crisis_points_Reinhart_Rogoff_All
    countries_to_sent.update(set(crisis_dict.keys()))


    # IMF program starts

    crisis_dict = crisis_points.imf_programs_monthly
    countries_to_sent.update(set(crisis_dict.keys()))

    crisis_dict = crisis_points.imf_programs_monthly_gap3
    countries_to_sent.update(set(crisis_dict.keys()))


    crisis_dict = crisis_points.imf_programs_monthly_gap6
    countries_to_sent.update(set(crisis_dict.keys()))


    # Remove completed countries - 60 base
    countries_to_sent = countries_to_sent - set(possible_countries)

    #args.countries = countries_to_sent

    args.countries = ['argentina']
    print(args.countries)

    time_df, uniq_periods = data_setup(doc_deetz, config.COUNTRY_FREQ_PERIOD)
    get_country_freqs_sample(args.countries, args.period, time_df, uniq_periods, args.out_dir, args.phraser,
                              filter_dict=None, word_defs=word_defs)