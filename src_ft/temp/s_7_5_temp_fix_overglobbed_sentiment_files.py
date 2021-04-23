"""
frequency_country_specific_freqs.py

Description: retrieve and save country-specific word frequencies for each supplied country. The word 
freq data for each country will only be based on articles which either mention the country name in the
title or abstract, or which are labeled with the region code corresponding to that particular country. 

usage: python3 frequency_country_specific_freqs.py
NOTE: can be done for as many countries at a time as you want.
"""
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
#from region_mapping import region
import os
import config
import pandas as pd
import numpy as np

def contains_ctry(value, ctry=None):
    return ctry in value

if __name__ == '__main__':

    sentiment_progress = pd.read_csv(os.path.join(config.AUG_DOC_META, 'sentiment_progress.csv'))
    countries = sentiment_progress['aug_doc_countries'].values

    #countries = ['argentina']

    in_dir = os.path.join(config.EVAL_WordDefs,'final_sent_merge')
    out_dir = os.path.join(config.EVAL_WordDefs,'final_sent_merge')

    issue_countries =  ['peru', 'mexico', 'turkey', 'israel', 'malaysia', 'norway', 'venezuela', 'finland', 'thailand', 'chile', 'uruguay', 'sweden', 'philippines', 'japan', 'iceland', 'denmark', 'bolivia', 'indonesia', 'spain', 'colombia', 'brazil', 'tanzania']

    for country in issue_countries:

        print('\nWorking on', country)
        in_file = os.path.join(in_dir, '{}_doc_sentiment_map.csv'.format(country))
        out_file = os.path.join(out_dir, '{}_doc_sentiment_map.csv'.format(country))

        sent_df = pd.read_csv(in_file).drop(columns='Unnamed: 0')
        sent_df = sent_df[sent_df['country'] == country]

        sent_df.to_csv(out_file)

