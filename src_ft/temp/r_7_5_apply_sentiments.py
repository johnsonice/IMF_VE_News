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

def sum_words(sentence, words):
    try:
        to_re = np.sum([sentence.count(x) for x in words])
    except:
        print('SENTENCE ', sentence, '\nWORDS', words)
        to_re = np.NaN
        exit(-1)
    return to_re
# Vader Sentiment analysis
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def vader_sentiment(sentence):
    return sid.polarity_scores(sentence)


def vader_postive(vader_rate):
    return vader_rate['pos']


def vader_negative(vader_rate):
    return vader_rate['neg']


def vader_is_postitive(vader_rate):
    return vader_rate['pos'] > vader_rate['neg']

# TextBlob sentiment analysis
from textblob import TextBlob

def testblob_rate(sentence):
    return TextBlob(sentence).sentiment


def textblob_polarity(testblob_rate):
    return testblob_rate.polarity


def textblob_is_positive(testblob_rate):
    return testblob_rate.polarity > 0


def textblob_subjectivity(testblob_rate):
    return testblob_rate.subjectivity


# flair sentiment analysis
'''
import flair
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

def flair_rate(sentence):
    s = flair.data.Sentence(sentence)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    return total_sentiment


def flair_is_positive(flair_rate):
    return flair_rate[0].value == 'POSITIVE'


def flair_pos_value(flair_rate):
    if flair_rate[0].value == 'POSITIVE':
        return flair_rate[0].score
    return 0


def flair_neg_value(flair_rate):
    if flair_rate[0].value == 'NEGATIVE':
        return flair_rate[0].score
    return 0
'''

# afinn sentiment analysis
from afinn import Afinn
af = Afinn()

def get_affin_score(sentence):
    return af.score(sentence)

def affin_is_positive(affin_score):
    return affin_score > 0


def get_sentiments(doc, word_defs):
    sent_df = pd.DataFrame(index=[0])
    sentence = ' '.join(doc)
    divisor = len(doc)

    # Word counts / sum words
    for def_name in word_defs.columns:
        sent_df[def_name] = sum_words(sentence, word_defs[def_name].dropna())/divisor

    # Vader
    vader_rate = vader_sentiment(sentence)
    sent_df['vader_pos'] = vader_postive(vader_rate)
    sent_df['vader_neg'] = vader_negative(vader_rate)

    # Testblob
    testblob_score = testblob_rate(sentence)
    sent_df['tb_polarity'] = textblob_polarity(testblob_score)
    sent_df['tb_is_positive'] = textblob_is_positive(testblob_score)
    sent_df['tb_subjectivity'] = textblob_subjectivity(testblob_score)

    '''
    # Flair
    flair_sent = flair_rate(sentence)
    sent_df['flair_is_positive'] = flair_is_positive(flair_sent)
    sent_df['flair_pos_value'] = flair_pos_value(flair_sent)
    sent_df['flair_neg_value'] = flair_neg_value(flair_sent)
    '''

    # Afinn
    af_score = get_affin_score(sentence)
    sent_df['afinn_score'] = af_score
    sent_df['affin_is_positive'] = affin_is_positive(af_score)

    return sent_df

def get_country_freqs_sample(countries, period_choice, time_df, uniq_periods, outdir, phraser, filter_dict=None,
                             sample_size=25, word_defs=None):

    # Get frequency data for each country supplied
    print("\nCounting Word Freqs...")

    # Save docname, country, month, sentiments, and word hitrate (count/sum words) according to different definitions
    huge_doc_map = pd.DataFrame()

    small_doc_map = None
    for country in countries:
        logger.info("\nWorking on {}".format(country))
        # for i, (period, doc_list) in enumerate(period_dict.items()):
        small_doc_map = None

        for i, period in enumerate(uniq_periods):
            #print("\r\tworking on period {} of {}...".format(i, len(uniq_periods)), end=' ')

            doc_list_a = country_period_filter(time_df, country, period)

            if len(doc_list_a) > sample_size:
                doc_list_a = np_choice(doc_list_a, size=sample_size)

            doc_list = [os.path.join(config.JSON_LEMMA, os.path.basename(p)) for p in doc_list_a]

            streamer = DocStreamer_fast(doc_list, language='en', phraser=phraser,
                                        stopwords=[], lemmatize=False).multi_process_files(workers=15, chunk_size=50)
            # count
            small_doc_map = pd.DataFrame(index=range(len(doc_list_a)))
            small_doc_map['doc_names'] = doc_list
            small_doc_map['month'] = period
            small_doc_map['country'] = country

            docnum = 0
            for doc in streamer:
                if doc is None:
                    continue

                sentiments = get_sentiments(doc, word_defs) # Returns DataFrame
                small_doc_map = pd.merge(small_doc_map, sentiments, left_index=True, right_index=True, how='outer')

        if small_doc_map is None:
            continue

        huge_doc_map = huge_doc_map.append(small_doc_map)

    outname = os.path.join(outdir, 'doc_sentiment_map.csv')
    huge_doc_map.to_csv(outname)
    return huge_doc_map

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
    word_defs = pd.read_csv(word_defs_f)

    class_type = 'Min1_AllCountry'
    doc_deetz = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(class_type))

    args.out_dir = config.EVAL_WordDefs
    #args.out_dir = os.path.join('/home/apsurek', 'pos_neg_compare')

    time_df, uniq_periods = data_setup(doc_deetz, config.COUNTRY_FREQ_PERIOD)
    get_country_freqs_sample(args.countries, args.period, time_df, uniq_periods, args.out_dir, args.phraser,
                              filter_dict=None, word_defs=word_defs)
