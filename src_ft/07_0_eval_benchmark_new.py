#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:48:38 2018

@author: chuang
"""

import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'./libs')
#sys.path.insert(0,'..')
import ujson as json
import config
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
from evaluate import get_recall,get_precision,get_fscore,get_eval_stats,get_preds_from_pd
from gensim.models.keyedvectors import KeyedVectors
from  crisis_points import crisis_points,ll_crisis_points
from mp_utils import Mp
import random
import statistics as stats

#%%
def get_stats(starts,ends,preds,offset,period,fbeta=2):
    tp, fn, mid_crisis  = [], [], []
    fq = period[0].lower()
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp() - offset, s.to_timestamp(), freq=fq), freq=fq)
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp(), freq=fq), freq=fq)
    
        period_tp = []
        # Collect True positives and preds happening during crisis
        for p in preds:
            if p in forecast_window: # True Positive if prediction occurs in forecast window
                period_tp.append(p)
            elif p in crisis_window: # if pred happened during crisis, don't count as fp
                mid_crisis.append(p)
    
        # Crisis counts as a false negative if no anomalies happen during forecast window
        if not any(period_tp): 
            fn.append(s)
        # True Positives for this crisis added to global list of TPs for the country
        tp += period_tp 
    
    # Any anomaly not occuring within forecast window (TPs) or happening mid-crisis is a false positive
    fp = set(preds) - set(tp) - set(mid_crisis)
    
    # Calc recall, precision, fscore
    recall = get_recall(len(tp), len(fn))
    precision = get_precision(len(tp), len(fp))
    fscore = get_fscore(len(tp), len(fp), len(fn), fbeta)
    
    return recall,precision,fscore

def get_country_vocab(country,period='quarter',frequency_path=config.FREQUENCY):
    data_path = os.path.join('/data/News_data_raw/FT_WD_research/frequency/temp/All_Comb','{}_{}_word_freqs.csv'.format(country, period))
    data = pd.read_pickle(data_path)
    vocab = list(data.index)
    return vocab

def collapse_scores(df):
    """
    use pandas to calculate aggregated f scores 
    """
    used_columns=['country','tp','fp','fn']
    agg_df = df[used_columns].groupby(['country']).sum()
    agg_df['recall'] = get_recall(agg_df['tp'], agg_df['fn'])
    agg_df['precision'] = get_precision(agg_df['tp'], agg_df['fp'])
    agg_df['f2'] = get_fscore(agg_df['tp'], agg_df['fp'], agg_df['fn'], beta=2)
    ## this is by country
    
    agg_df_all = df[['tp','fp','fn']].sum()
    agg_df_all['recall'] = get_recall(agg_df_all['tp'], agg_df_all['fn'])
    agg_df_all['precision'] = get_precision(agg_df_all['tp'], agg_df_all['fp'])
    agg_df_all['f2'] = get_fscore(agg_df_all['tp'], agg_df_all['fp'], agg_df_all['fn'], beta=2)
    
    return agg_df,agg_df_all

#%%
if __name__ == "__main__":
    period = config.COUNTRY_FREQ_PERIOD
    vecs = KeyedVectors.load(config.W2V)
    vocabs =list(vecs.wv.vocab.keys())

    
    def get_country_stats(country,vocabs=vocabs,period=period,n_words=50,n_iter=100,crisis_defs=config.crisis_defs):
        fq = period[0].lower()
        country_stats = list()
        c_vocab = get_country_vocab(country,period=period)
        for seed in range(0,n_iter):
            random.seed(seed)
            word_groups = random.sample(vocabs,n_words*5)
            word_groups = [w for w in word_groups if w in c_vocab][:n_words]
            df = aggregate_freq(word_groups, 
                                country,
                                period=period,
                                stemmed=False,
                                frequency_path=config.FREQUENCY)
            
            ## make data ends at when crisis data ends 
            df = df[:config.eval_end_date[fq]]
            
            if seed == 0: 
                print(country)
                print(df.head(2))
            #offset = pd.DateOffset(months=config.months_prior)
            
            if crisis_defs == 'kr':
                starts = list(pd.PeriodIndex(crisis_points[country]['starts'], freq=fq))
                ends = list(pd.PeriodIndex(crisis_points[country]['peaks'], freq=fq))
            elif crisis_defs == 'll':
                starts = list(pd.PeriodIndex(ll_crisis_points[country]['starts'], freq=fq))
                ends = list(pd.PeriodIndex(ll_crisis_points[country]['peaks'], freq=fq))
                
            #preds = list(signif_change(df, window=config.smooth_window_size, direction='incr').index)
            preds = get_preds_from_pd(df,
                          country,
                          method='zscore', 
                          crisis_defs=crisis_defs,
                          period=period, 
                          window=config.smooth_window_size, 
                          direction='incr', 
                          months_prior=config.months_prior, 
                          fbeta=2,
                          weights=None,
                          z_thresh=config.z_thresh)
            res_stats = list(get_eval_stats(fq,starts,ends,preds,period,months_prior=config.months_prior,fbeta=2))
            res_stats.insert(0,country)

            country_stats.append(res_stats)
            #recall, precision, fscore, len(tp), len(fp), len(fn)
            
            #recall,precision,fscore = get_stats(starts,ends,preds,offset,period)
            #print(recall,precision,fscore)
            #country_stats.append(res_stats)
            
        
        return country_stats
    
    #res = [average_score(c) for c in config.countries[:5]]
    
    mp = Mp(config.countries,get_country_stats)
    overall_res = mp.multi_process_files(workers=5,chunk_size=1)
    overall_res = [r for rl in overall_res for r in rl]
    
    res_df = pd.DataFrame(overall_res,columns=['country','recall','precision','f2','tp','fp','fn'])
    ## aggregate by country 
    country_df,agg_df_all = collapse_scores(res_df)
    #out_csv = os.path.join(config.EVAL, '{}_offset_{}_smooth_{}_ramdom_sampling_bench_mark_TEST.csv'.format(period,config.months_prior,config.smooth_window_size))
    out_csv = os.path.join(config.EVAL, '{}_offset_{}_smooth_{}_ramdom_sampling_bench_mark_TEST.csv'.format(period,config.months_prior,config.smooth_window_size))
    agg_df_all.to_csv(out_csv)
    print(agg_df_all)
    #final_df.to_csv('temp_data/agg_tuning_res.csv')
#%%
#    df = pd.DataFrame(res,columns=['country','fscores','mean','std'])
#    out_csv = os.path.join(config.EVAL, '{}_offset_{}_smooth_{}_ramdom_sampling_bench_mark.csv'.format(period,config.months_prior,config.smooth_window_size))
#    df.to_csv(out_csv)
#    
#    print(df['mean'].mean())

    