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

    fine_countries =  ['pakistan', 'united-states', 'new-zealand', 'latvia', 'poland', 'ukraine', 'russia', 'kenya', 'australia', 'jordan', 'canada', 'egypt', 'ireland', 'nigeria', 'zambia', 'france', 'nicaragua', 'zimbabwe', 'greece', 'india', 'south-africa', 'tunisia', 'jamaica', 'ecuador', 'vietnam', 'united-kingdom', 'belgium', 'uganda', 'italy', 'china', 'south-korea', 'germany', 'bulgaria', 'argentina', 'lebanon', 'netherlands', 'hungary']

    countries = list(set(countries)-set(fine_countries))

    #countries = ['argentina']

    in_dir = os.path.join(config.EVAL_WordDefs,'final_sent_merge')
    class_type = 'Min1_AllCountry'
    doc_deetz = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(class_type))
    aug_doc = pd.read_pickle(doc_deetz)
    aug_doc = aug_doc[aug_doc['country_n'] > 0]

    orig_issues =  ['peru', 'mexico', 'turkey', 'israel', 'malaysia', 'norway', 'venezuela', 'finland', 'thailand', 'chile', 'uruguay', 'sweden', 'philippines', 'japan', 'iceland', 'denmark', 'bolivia', 'indonesia', 'spain', 'colombia', 'brazil', 'tanzania']

    issue_countries = []

    for country in countries:

        print('\nWorking on', country)
        in_file = os.path.join(in_dir, '{}_doc_sentiment_map.csv'.format(country))

        sent_df = pd.read_csv(in_file).drop(columns='Unnamed: 0')
        count_aug = aug_doc['country'].apply(contains_ctry, ctry=country).sum()
        count_sent = len(sent_df)

        if count_aug == count_sent:
            print('Counts for {} match up'.format(country))
            fine_countries.append(country)
        elif count_aug > count_sent:
            print('Count in aug doc bigger for {}'.format(country))
            print('\tAug: {}, Sent: {}'.format(count_aug, count_sent))
            issue_countries.append(country)
        else:
            print('Count in sent bigger for {}'.format(country))
            print('\tAug: {}, Sent: {}'.format(count_aug, count_sent))
            issue_countries.append(country)


    print('Fine countries\n',fine_countries)
    print('\nIssue countries\n',issue_countries)
    print('\n\nPrevious issue count: {}\nNew issue count: {}'.format(len(orig_issues), len(issue_countries)))