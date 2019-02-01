"""
frequency_eval.py

Description: Used to evaluate supplied terms and term groups wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval.py <TERM1> <TERM2> ...
NOTE: to see an explanation of optional arguments, use python3 frequency_eval.py --help
"""
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
#import argparse
from gensim.models.keyedvectors import KeyedVectors
from crisis_points import crisis_points
from frequency_utils import aggregate_freq, signif_change
from evaluate import get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats,get_eval_stats,get_preds_from_pd
import pandas as pd
import numpy as np
import os
from mp_utils import Mp
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
        try:
            words, weights = get_input_words_weights(args,
                                                 w,
                                                 vecs=vecs,
                                                 weighted=args.weighted)
            sim_word_group.extend(words)
        except:
            print('Not in vocabulary {}'.format(w))
    sim_word_set = set(sim_word_group)
    return sim_word_set

def eval_word_group(countries,args,word_list,weights=None,z_threshs=[1,2]):
    fq = args.period[0].lower()
    results = list()
    for country in countries:
        ag_freq = aggregate_freq(word_list, country, args.period,False,args.frequency_path,weights=weights)
        ## make data ends at when crisis data ends 
        ag_freq = ag_freq[:config.eval_end_date[fq]]
        for z in z_threshs:
            if not isinstance(ag_freq, pd.Series):
                print('\nno data for {}\n'.format(country))
                results.append((country,z,np.nan, np.nan, np.nan))
            #offset = pd.DateOffset(months=config.months_prior)
            starts = list(pd.PeriodIndex(crisis_points[country]['starts'], freq=fq))
            ends = list(pd.PeriodIndex(crisis_points[country]['peaks'], freq=fq))
            #preds = list(signif_change(ag_freq, window=config.smooth_window_size, direction='incr',z_thresh=z).index)
            preds = get_preds_from_pd(ag_freq,
                                      country,
                                      method=args.method, 
                                      crisis_defs=args.crisis_defs,
                                      period=args.period, 
                                      window=args.window, 
                                      direction='incr', 
                                      months_prior=args.months_prior, 
                                      fbeta=2,
                                      weights=None,
                                      z_thresh=z)
            
            stats = list(get_eval_stats(fq,starts,ends,preds,args.period,months_prior=config.months_prior,fbeta=2))
            #recall, precision, fscore, len(tp), len(fp), len(fn)
            meta = [country,z]
            meta.extend(stats)
            results.append(meta)

    ## resutls is a list of list of scores
    return results

def collapse_scores(df,group_by='language'):
    """
    use pandas to calculate aggregated f scores 
    """
    used_columns=['language','thresh','tp','fp','fn']
    agg_df = df[used_columns].groupby(['language','thresh']).sum()
    agg_df['recall'] = get_recall(agg_df['tp'], agg_df['fn'])
    agg_df['precision'] = get_precision(agg_df['tp'], agg_df['fp'])
    agg_df['f2'] = get_fscore(agg_df['tp'], agg_df['fp'], agg_df['fn'], beta=2)
    return agg_df


class args_class(object):
    def __init__(self, targets,frequency_path=config.FREQUENCY,eval_path=config.EVAL_WG,
                 wv_path = config.W2V,topn=config.topn,months_prior=config.months_prior,
                 window=config.smooth_window_size,
                 countries=config.countries,
                 period=config.COUNTRY_FREQ_PERIOD,
                 eval_end_date=config.eval_end_date,
                 method='zscore',crisis_defs='kr',
                 sims=True,weighted=False,z_thresh=config.z_thresh):
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
        self.z_thresh= z_thresh
#%%
if __name__ == '__main__':
    
    ## load config arguments
    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,
                          sims=config.SIM,period=config.COUNTRY_FREQ_PERIOD, 
                          months_prior=config.months_prior,
                          window=config.smooth_window_size,
                          eval_end_date=config.eval_end_date,
                          weighted= config.WEIGHTED,
                          z_thresh=config.z_thresh)

    # Parse input word groups, word_gropus is a list of list:
    # something like this: [['fear'],['worry'],['concern'],['risk'],['threat'],['warn'],['maybe']]
    file_path = os.path.join(config.SEARCH_TERMS,config.GROUPED_SEARCH_FILE)
    search_groups = read_grouped_search_words(file_path)  
    ## it is a dictionary list:
#       {'fear_language': [['fear']],
#       'risk_language': [['threat'], ['warn']]}

    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)
        
    if args.weighted:   
        raise Exception('for now, this module only works for unweighted calculation')
        print('Weighted flag = True ; Results are aggregated by weighted sum....')
    else:
        search_words_sets = dict()
        for k,v in search_groups.items():
            if args.sims:
                search_words_sets[k]=list(get_sim_words_set(args,search_groups[k])) ## turn set to list
            else:
                search_words_sets[k] = [t for tl in v for t in tl] ## flattern the list of list 
        weights = None
    
    #print(search_words_sets)

    # Get prec, rec, and fscore for each country for each word group
    multi_process = True
    iter_items = list(search_words_sets.items())
    z_thresh_range = np.arange(1,4,0.1)
    
    if multi_process :
        # run in multi process
        def multi_run_eval_thresh(item,args=args,weights=None,z_threshs=z_thresh_range):
            k,word_list = item
            res_stats = eval_word_group(args.countries,args,word_list,weights=weights,z_threshs=z_thresh_range)
            [r.insert(0,k) for r in res_stats]
            return res_stats
        
        mp = Mp(iter_items,multi_run_eval_thresh)
        overall_res = mp.multi_process_files(workers=5, chunk_size=1)  ## do not set workers to be too high, your memory will explode
        overall_res = [res for res_list in overall_res for res in res_list]
    else:
        print('Running in 1 process, will take long time ....')
        #args.countries=args.countries[:1]
        overall_res = list()
        for item in iter_items:
            print(k)
            res_list = list()
            k,word_list = item
            res_list = eval_word_group(args.countries,args,word_list,weights=weights,z_threshs=z_thresh_range)
            #print(res_list)
            [r.insert(0,k) for r in res_list]
            overall_res.extend(res_list)
    
    
    res_df = pd.DataFrame(overall_res,columns=['language','country','thresh','recall','precision','f2','tp','fp','fn'])
    res_df.to_csv('temp_data/all_thresh_tuning.csv')
    ## aggregate by country 
    final_df = collapse_scores(res_df)
    final_df.to_csv('temp_data/agg_tuning_res.csv')

