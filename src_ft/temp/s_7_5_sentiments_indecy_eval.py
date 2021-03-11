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


def apply_expansions(df, base=('fed', 'w2v', 'w2v_refined_2')):
    df = df.copy()

    df['vader_is_pos'] = df['vader_pos'] > df['vader_neg']
    for b_col in base:
        b_pos = b_col + '_pos'
        b_neg = b_col + '_neg'

        # Vaders
        df['vader_pos_x_{}'.format(b_pos)] = df['vader_pos'] * df[b_pos]
        df['vader_neg_x_{}'.format(b_neg)] = df['vader_neg'] * df[b_neg]
        df['vader_is_pos_x_{}'.format(b_pos)] = df['vader_is_pos'] * df[b_pos]
        df['vader_is_neg_x_{}'.format(b_neg)] = (1 - df['vader_is_pos']) * df[b_neg]

        # TextBlob
        df['textblob_polar_x_{}'.format(b_pos)] = df['tb_is_positive'] * df['tb_polarity'].apply(abs) * df[b_pos]
        df['textblob_polar_x_{}'.format(b_neg)] = (1 - df['tb_is_positive']) * df['tb_polarity'].apply(abs) * df[b_neg]
        df['textblob_is_pos_x_{}'.format(b_pos)] = df['tb_is_positive'] * df[b_pos]
        df['textblob_is_neg_x_{}'.format(b_neg)] = (1 - df['tb_is_positive']) * df[b_neg]

        # Afinn
        df['afinn_score_x_{}'.format(b_pos)] = df['affin_is_positive'] * df['afinn_score'].apply(abs) * df[b_pos]
        df['afinn_score_x_{}'.format(b_neg)] = (1 - df['affin_is_positive']) * df['afinn_score'].apply(abs) * df[b_neg]
        df['afinn_is_pos_x_{}'.format(b_pos)] = df['affin_is_positive'] * df[b_pos]
        df['afinn_is_neg_x_{}'.format(b_neg)] = (1 - df['affin_is_positive']) * df[b_neg]

    return df


def plot_and_correlate_pairs(expanded_df):
    correlations = pd.DataFrame(index=['Correlation'])

    definitions = {
        # Base Counts
        'Fed Words': ('fed_pos', 'fed_neg'),
        'w2v Words': ('w2v_pos', 'w2v_neg'),
        'w2v Words Revised': ('w2v_refined_0_pos', 'w2v_refined_0_neg'),
        'w2v Words Revised v2': ('w2v_refined_1_pos', 'w2v_refined_1_neg'),
        'w2v Words Revised v3': ('w2v_refined_2_pos', 'w2v_refined_2_neg'),

        # Fed words x Sentiments
        'Vader Sentiment Score': ('vader_pos', 'vader_neg'),
        'Vader Score x Fed Words': ('vader_pos_x_fed_pos', 'vader_neg_x_fed_neg'),
        'Vader Direction x Fed Words': ('vader_is_pos_x_fed_pos', 'vader_is_neg_x_fed_neg'),
        'TextBlob Score x Fed Words': ('textblob_polar_x_fed_pos', 'textblob_polar_x_fed_neg'),
        'TextBlob Direction x Fed Words': ('textblob_is_pos_x_fed_pos',
                                           'textblob_is_neg_x_fed_neg'),
        'Afinn Score x Fed Words': ('afinn_score_x_fed_pos', 'afinn_score_x_fed_neg'),
        'Afinn Direction x Fed Words': ('afinn_is_pos_x_fed_pos', 'afinn_is_neg_x_fed_neg'),

        # w2v words x Sentiments
        'Vader Score x w2v Words': ('vader_pos_x_w2v_pos', 'vader_neg_x_w2v_neg'),
        'Vader Direction x w2v Words': ('vader_is_pos_x_w2v_pos', 'vader_is_neg_x_w2v_neg'),
        'Textblob Score x w2v Words': ('textblob_polar_x_w2v_pos', 'textblob_polar_x_w2v_neg'),
        'Textblob Direction x w2v Words': ('textblob_is_pos_x_w2v_pos',
                                           'textblob_is_neg_x_w2v_neg'),
        'Afinn Score x w2v Words': ('afinn_score_x_w2v_pos', 'afinn_score_x_w2v_neg'),
        'Afinn Direction x w2v Words': ('afinn_is_pos_x_w2v_pos', 'afinn_is_neg_x_w2v_neg'),

        # w2v revised words x Sentiments
        'Vader Score x w2v Words Revised v3': ('vader_pos_x_w2v_refined_2_pos',
                                               'vader_neg_x_w2v_refined_2_neg'),
        'Vader Direction x w2v Words Revised v3': ('vader_is_pos_x_w2v_refined_2_pos',
                                                   'vader_is_neg_x_w2v_refined_2_neg'),
        'TextBlob Score x w2v Words Revised v3': ('textblob_polar_x_w2v_refined_2_pos',
                                                  'textblob_polar_x_w2v_refined_2_neg'),
        'TextBlob Direction x w2v Words Revised v3': ('textblob_is_pos_x_w2v_refined_2_pos',
                                                      'textblob_is_neg_x_w2v_refined_2_neg'),
        'Afinn Score x w2v Words Revised v3': ('afinn_score_x_w2v_refined_2_pos',
                                               'afinn_score_x_w2v_refined_2_neg'),
        'Afinn Direction x w2v Words Revised v3': ('afinn_is_pos_x_w2v_refined_2_pos',
                                                   'afinn_is_neg_x_w2v_refined_2_neg')
    }

    for key in definitions.keys():
        title = str(key)
        pos_t = definitions[key][0]
        pos_v = expanded_df[pos_t]
        neg_t = definitions[key][1]
        neg_v = expanded_df[neg_t]
        corr = np.corrcoef(pos_v.values, neg_v.values)[0]
        correlations[title] = corr
        #graph_and_correlate(pos_v, neg_v, True, key)

    return correlations.T


#in_file = os.path.join(config.EVAL_WordDefs, 'doc_sentiment_map.csv')
in_file = os.path.join(config.EVAL_WordDefs, 'doc_sentiment_map_test.csv')

in_df = pd.read_csv(in_file)

expanded = apply_expansions(in_df)
grouped = expanded.groupby(['country','month']).mean()

#out_file = os.path.join(config.EVAL_WordDefs, 'month_sentiment_indeces.csv')
out_file = os.path.join(config.EVAL_WordDefs, 'month_sentiment_indeces_test.csv')

grouped.to_csv(out_file)

#corr_file = os.path.join(config.EVAL_WordDefs, 'corr_sentiment_indeces.csv')
corr_file = os.path.join(config.EVAL_WordDefs, 'corr_sentiment_indeces_test.csv')

corr_df = plot_and_correlate_pairs(grouped)
corr_df.to_csv(corr_file)

