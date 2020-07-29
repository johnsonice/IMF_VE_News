#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:28:33 2018

@author: chuang
"""

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
from evaluate import get_recall,get_precision,get_fscore
from gensim.models.keyedvectors import KeyedVectors
from  crisis_points import crisis_points
from mp_utils import Mp
import random
import statistics as stats
import logging
f_handler = logging.FileHandler('/home/apsurek/logs/err_log_6_01.log')
f_handler.setLevel(logging.WARNING)


#%%
def get_stats(starts,ends,preds,offset,fbeta=2):
    tp, fn, mid_crisis  = [], [], []
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp() - offset, s.to_timestamp(), freq='q'), freq='q')
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp(), freq='q'), freq='q')
    
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

#def get_country_vocab(country,period='quarter',frequency_path=config.FREQUENCY):
#    data_path = os.path.join(frequency_path,'{}_{}_word_freqs.pkl'.format(country, period))
#    data = pd.read_pickle(data_path)
#    vocab = list(data.index)
#    return vocab

def get_sim_words(vecs,wg,topn):
    if not isinstance(wg,list): 
        wg = [wg]
    try:
        sims = [w[0] for w in vecs.wv.most_similar(wg, topn=topn)]
    except KeyError:
        try:
            wg_update = list()
            for w in wg:
                wg_update.extend(w.split('_'))
            sims = [w[0] for w in vecs.wv.most_similar(wg_update, topn=topn)]
            print('Warning: {} not in the vocabulary, split the word with _'.format(wg))
        except:
            print('Not in vocabulary: {}'.format(wg_update))
            return wg
    words = sims + wg
    return words


#%%
if __name__ == "__main__":
    period = config.COUNTRY_FREQ_PERIOD
    vecs = KeyedVectors.load(config.W2V)
    
    def export_country_ts(country,period=period,vecs=vecs):
        series_wg = list()
        for wg in config.targets:
            word_groups = get_sim_words(vecs,wg,15)
            df = aggregate_freq(word_groups, country,period=period,stemmed=False,frequency_path=config.FREQUENCY)
            df.name = wg
            series_wg.append(df)
        
        df_all = pd.concat(series_wg,axis=1)
        out_csv = os.path.join(config.EVAL_TS, '{}_{}_time_series.csv'.format(country,period))
        df_all.to_csv(out_csv)
        
        return country, df_all


    class_type = None

    frequency_path = config.FREQUENCY

    class_type_setups = config.class_type_setups

    for setup in class_type_setups:

        class_type = setup[0]
        freq_path = os.path.join(frequency_path, class_type)  # Moved the TF_DFs manually for speed since 06_0

        def export_country_ts_2(country, period=period, vecs=vecs, class_type=class_type, frequency_path=freq_path):
            series_wg = list()
            for wg in config.targets:
                word_groups = get_sim_words(vecs, wg, 15)
                df = aggregate_freq(word_groups, country, period=period, stemmed=False, frequency_path=frequency_path)
                df.name = wg
                series_wg.append(df)

            df_all = pd.concat(series_wg, axis=1)
            out_csv = os.path.join(config.EVAL_TS, class_type, '{}_{}_time_series.csv'.format(country, period))
            df_all.to_csv(out_csv)

            return country, df_all


        countries = config.countries
        mp = Mp(countries, export_country_ts_2)
        res = mp.multi_process_files(chunk_size=1)
