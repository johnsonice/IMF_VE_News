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
import crisis_points


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
        #'w2v Words Revised': ('w2v_refined_0_pos', 'w2v_refined_0_neg'),
        #'w2v Words Revised v2': ('w2v_refined_1_pos', 'w2v_refined_1_neg'),
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
        corr = np.corrcoef(pos_v.values, neg_v.values)[0][1]
        correlations[title] = corr
        #graph_and_correlate(pos_v, neg_v, True, key)

    return correlations.T

if __name__ == '__main__':

    sentiment_progress = pd.read_csv(os.path.join(config.AUG_DOC_META, 'sentiment_progress.csv'))
    done_countries = sentiment_progress['aug_doc_countries'].values

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
    countries_to_sent = countries_to_sent - set(done_countries)

    possible_countries = countries_to_sent


    #countries = ['argentina']

    in_dir = os.path.join(config.EVAL_WordDefs,'final_sent_merge_new')
    out_dir = os.path.join(config.EVAL_WordDefs,'final_sent_mean2_new')
    corr_dirr = os.path.join(config.EVAL_WordDefs,'month_sentiment_correlations')

    for country in possible_countries:
        #in_file = os.path.join(config.EVAL_WordDefs, 'doc_sentiment_map.csv')
        print(f"Working on {country}")

        in_file = os.path.join(in_dir, '{}_doc_sentiment_map.csv'.format(country))

        in_df = pd.read_csv(in_file).drop(columns='Unnamed: 0')

        if in_df.empty:
            print(f'\tEmpty frame (no observations) for {country}')
            continue

        expanded = apply_expansions(in_df)
        grouped = expanded.groupby(['country','month']).mean()

        #out_file = os.path.join(config.EVAL_WordDefs, 'month_sentiment_indeces.csv')
        out_file = os.path.join(out_dir, '{}_month_sentiment_indeces.csv'.format(country))
        grouped.to_csv(out_file)
        print('Saved indecies for', country)

        #corr_file = os.path.join(config.EVAL_WordDefs, 'corr_sentiment_indeces.csv')
        corr_file = os.path.join(corr_dirr, '{}_corr_sentiment_indeces.csv'.format(country))

        corr_df = plot_and_correlate_pairs(grouped)
        corr_df.to_csv(corr_file)
        print('Saved correlations for', country)


