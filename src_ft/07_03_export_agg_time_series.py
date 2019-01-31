"""
frequency_eval.py

Description: Used to evaluate supplied terms and term groups wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval.py <TERM1> <TERM2> ...
NOTE: to see an explanation of optional arguments, use python3 frequency_eval.py --help
"""
import sys
sys.path.insert(0,'./libs')
#import argparse
from gensim.models.keyedvectors import KeyedVectors
from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
import pandas as pd
import numpy as np
from mp_utils import Mp
import os
import config

#%%
def read_grouped_search_words(file_path):
    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k,v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]
        temp_list = [wg.split('&') for wg in temp_list]   ## split & for wv search 
        search_groups[k]=temp_list
    return search_groups

def get_sim_words_set(args,word_group):
    assert isinstance(word_group,list)     
    sim_word_group = list()
    for w in word_group:
        words, weights = get_input_words_weights(args,
                                             w,
                                             vecs=vecs,
                                             weighted=args.weighted)
        sim_word_group.extend(words)
    sim_word_set = set(sim_word_group)
    return sim_word_set
    
def get_crisis_wondiws(args,crisis_points,country):
    
    starts = list(pd.PeriodIndex(crisis_points[country]['starts'], freq=args.period[0]))
    ends = list(pd.PeriodIndex(crisis_points[country]['peaks'], freq=args.period[0]))
    
    crisis_windows=None
    for s, e in zip(starts, ends):
        if crisis_windows is None:
            crisis_windows = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp()), freq=args.period[0])
        else:
            crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp()), freq=args.period[0])
            crisis_windows= crisis_windows.append(crisis_window)
    
    ## drop duplicates
    crisis_windows=crisis_windows.drop_duplicates()
    
    ## create crisis df for merging
    crisis_df = pd.DataFrame(np.ones(len(crisis_windows)),index=crisis_windows,columns=['crisis_window'])
    crisis_df.sort_index(inplace=True)
    
    return crisis_df

def get_bop_crisis(args,crisis_points,country):
    
    bop_crisis = list(pd.PeriodIndex(crisis_points[country]['bop'], freq=args.period[0]))
    if len(bop_crisis) == 0: 
        return None
    else:
        bop_crisis_df = pd.DataFrame(np.ones(len(bop_crisis)),index=bop_crisis,columns=['bop_crisis'])
        bop_crisis_df.sort_index(inplace=True)
    
    return bop_crisis_df

class args_class(object):
    def __init__(self, targets,frequency_path=config.FREQUENCY,eval_path=config.EVAL_WG,
                 wv_path = config.W2V,topn=config.topn,months_prior=config.months_prior,
                 window=config.smooth_window_size,
                 countries=config.countries,
                 period=config.COUNTRY_FREQ_PERIOD,
                 eval_end_date=config.eval_end_date,
                 method='zscore',crisis_defs='kr',sims=True,weighted=False,
                 z_thresh=config.z_thresh):
        self.targets = targets
        self.frequency_path = frequency_path
        self.eval_path=eval_path
        self.wv_path = wv_path
        self.topn = topn
        self.months_prior = months_prior
        self.window = window
        self.countries = countries
        self.method = method
        self.period = period
        self.eval_end_date=eval_end_date
        self.crisis_defs = crisis_defs
        self.sims = sims
        self.weighted = weighted
        self.z_thresh=z_thresh
#%%
#args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
#              countries = config.countries,wv_path = config.W2V,
#              sims=True,period=config.COUNTRY_FREQ_PERIOD, 
#              months_prior=config.months_prior,
#              window=config.smooth_window_size,
#              eval_end_date=config.eval_end_date,
#              weighted= config.WEIGHTED)
#
#test = get_crisis_wondiws(args,crisis_points,'argentina')
        
        #%%
if __name__ == '__main__':
    
    ## load config arguments
    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,
                          sims=True,period=config.COUNTRY_FREQ_PERIOD, 
                          months_prior=config.months_prior,
                          window=config.smooth_window_size,
                          eval_end_date=config.eval_end_date,
                          weighted= config.WEIGHTED,
                          z_thresh=config.z_thresh)

    # Parse input word groups, word_gropus is a list of list:
    # something like this: [['fear'],['worry'],['concern'],['risk'],['threat'],['warn'],['maybe']]
    file_path = os.path.join(config.SEARCH_TERMS,config.GROUPED_SEARCH_FILE)  ## searh words name
    search_groups = read_grouped_search_words(file_path)  
    ## it is a dictionary list:
#       {'fear_language': ['fear'],
#       'risk_language': ['threat', 'warn'],
#       'hedging_language': ['could', 'perhaps', 'may', 'possibly', 'uncertain'],
#       'opinion_language': ['say', 'predict', 'tell', 'believe'],
#       'risis_language': ['financial_crisis', 'depression']}
    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)
        
    if args.weighted:   
        raise Exception('for now, this module only works for unweighted calculation')
        print('Weighted flag = True ; Results are aggregated by weighted sum....')
    else:
        search_words_sets = dict()
        for k,v in search_groups.items():
            search_words_sets[k]=list(get_sim_words_set(args,search_groups[k])) ## turn set to list
        weights = None
    
    def export_country_ts(country,search_words_sets=search_words_sets,period=args.period,z_thresh=args.z_thresh,export=True):
        series_wg = list()
        for k,words in search_words_sets.items(): 
            word_groups = words
            df = aggregate_freq(word_groups, country,period=period,stemmed=False,
                                frequency_path=args.frequency_path,
                                weights=None)
            df.name = k
            
            preds = list(signif_change(df, 
                                   args.window, 
                                   period=args.period,
                                   direction='incr',
                                   z_thresh=z_thresh).index) 
            pred_df = pd.DataFrame(np.ones(len(preds)),index=preds,columns=[k+'_pred_crisis'])
            df = pred_df.join(df,how='right')
            series_wg.append(df)

        df_all = pd.concat(series_wg,axis=1)
        crisis_df = get_crisis_wondiws(args,crisis_points,country)
        bop_crisis_df = get_bop_crisis(args,crisis_points,country)
        
        ## merge crisis events 
        df_all=df_all.join(crisis_df)
        
        if bop_crisis_df is None:
            df_all['bop_crisis'] = 0 
        else:
            df_all=df_all.join(bop_crisis_df)
            
        df_all.fillna(0,inplace=True)
        
        if export:
            out_csv = os.path.join(config.EVAL_TS, 'agg_{}_{}_z{}_time_series.csv'.format(country,period))
            df_all.to_csv(out_csv)
        
        return country,df_all
        # end of function 
        
#    country = "argentina"
#    c,d = export_country_ts(country)

    mp = Mp(config.countries,export_country_ts)
    res = mp.multi_process_files(chunk_size=1,workers=10)
    



        
