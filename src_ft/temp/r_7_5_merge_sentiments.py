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
import shutil

if __name__ == '__main__':

    sentiment_progress = pd.read_csv(os.path.join(config.AUG_DOC_META, 'sentiment_progress.csv'))
    #possible_countries = sentiment_progress['aug_doc_countries'].values

    in_dir = os.path.join(config.EVAL_WordDefs,'final_sent2')
    out_dir = os.path.join(config.EVAL_WordDefs,'final_sent_merge')

    possible_countries = ['japan', 'tanzania']

    for cntry in possible_countries:
        cntry_fbase = os.path.join(in_dir, cntry+'*')
        cntry_files = glob.glob(cntry_fbase)
        if len(cntry_files) == 0:
            print(cntry + ' HAS NO FILES!!!')
        elif len(cntry_files) == 1:
            shutil.copy(cntry_files[0], out_dir)
            print('Copied over '+cntry)
        else:
            out_f = os.path.join(out_dir, '{}_doc_sentiment_map.csv'.format(cntry))
            merge_df = pd.read_csv(cntry_files[0]).drop(columns='Unnamed: 0')
            for c_file in cntry_files[1:]:
                c_df = pd.read_csv(c_file).drop(columns='Unnamed: 0')
                merge_df = pd.concat([merge_df, c_df])
            merge_df.to_csv(out_f)
            print('Merged and saved ' + cntry)
