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
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import pandas as pd
import config
import glob

if __name__ == '__main__':

    sentiment_progress = pd.read_csv(os.path.join(config.AUG_DOC_META, 'sentiment_progress.csv'))
    possible_countries = sentiment_progress['aug_doc_countries'].values

    in_dir = os.path.join(config.EVAL_WordDefs,'final_sent_merge_new')
    out_dir = os.path.join(config.EVAL_WordDefs,'final_sent_mean_new')
    out_f = os.path.join(out_dir, '{}_month_sentiment.csv')

    for cntry in possible_countries:
        cntry_fbase = os.path.join(in_dir, cntry+'*')
        cntry_files = glob.glob(cntry_fbase)
        if len(cntry_files) == 0:
            print(cntry + ' HAS NO FILES!!!')
        else:
            c_df = pd.read_csv(cntry_files[0]).drop(columns='Unnamed: 0')
            mean_df = c_df.groupby('month').mean()
            mean_df.to_csv(out_f.format(cntry))
            print('Averaged and saved ' + cntry)
