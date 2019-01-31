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
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
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

def eval_word_group(countries,word_list,period,stemmed,frequency_path,weights=None,z_thresh=1.96):
    fq = period[0].lower()
    results = list()
    for country in countries:
        df = aggregate_freq(word_list, country, period, stemmed,frequency_path,weights=weights)
        if not isinstance(df, pd.Series):
            print('\nno data for {}\n'.format(country))
            results.append((np.nan, np.nan, np.nan))
        ## make data ends at when crisis data ends 
        df = df[:config.eval_end_date[fq]]
        offset = pd.DateOffset(months=config.months_prior)
        starts = list(pd.PeriodIndex(crisis_points[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_points[country]['peaks'], freq=fq))
        preds = list(signif_change(df, window=config.smooth_window_size, direction='incr',z_thresh=z_thresh).index)
        temp_res = get_stats(starts,ends,preds,offset,period)
        # recall,precision,fscore,len(tp), len(fp), len(fn)
        results.append(temp_res)
    
    ## calculate average   
    #print(results)
    all_stats = pd.DataFrame(results,columns=['recall','precision','fscore','tp', 'fp', 'fn'])
    tp, fp, fn = all_stats['tp'].sum(), all_stats['fp'].sum(), all_stats['fn'].sum()
    recall = get_recall(tp, fn)
    prec = get_precision(tp, fp)
    f2 = get_fscore(tp, fp, fn, beta=2)
#    avg = pd.Series([recall, prec, f2, tp, fp, fn], 
#                    name='aggregate', 
#                    index=['recall','precision','fscore','tp','fp','fn'])
#    all_stats = all_stats.append(avg)
    
    return recall,prec,f2  ## recall, precision, fscore


def get_stats(starts,ends,preds,offset,period,fbeta=2):
    tp, fn, mid_crisis  = [], [], []
    fq = period[0].lower()
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(how='s') - offset, s.to_timestamp(how='s'), freq=fq), freq=fq)
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(how='s'), e.to_timestamp(how='e'), freq=fq), freq=fq)

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
    print(recall,precision,fscore,len(tp), len(fp), len(fn))
    return recall,precision,fscore,len(tp), len(fp), len(fn)


#    tp, fn, mid_crisis  = [], [], []
#    for s, e in zip(starts, ends):
#        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp() - offset, s.to_timestamp()), freq=fq)
#        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp()), freq=fq)
#
#        period_tp = []
#        # Collect True positives and preds happening during crisis
#        for p in preds:
#            if p in forecast_window: # True Positive if prediction occurs in forecast window
#                period_tp.append(p)
#            elif p in crisis_window: # if pred happened during crisis, don't count as fp
#                mid_crisis.append(p)
#
#        # Crisis counts as a false negative if no anomalies happen during forecast window
#        if not any(period_tp): 
#            fn.append(s)
#        # True Positives for this crisis added to global list of TPs for the country
#        tp += period_tp 
#
#    # Any anomaly not occuring within forecast window (TPs) or happening mid-crisis is a false positive
#    fp = set(preds) - set(tp) - set(mid_crisis)
#
#    # Calc recall, precision, fscore
#    recall = get_recall(len(tp), len(fn))
#    precision = get_precision(len(tp), len(fp))
#    fscore = get_fscore(len(tp), len(fp), len(fn), fbeta)
#    
#    print(recall, precision, fscore, len(tp), len(fp), len(fn))
#    return recall, precision, fscore, len(tp), len(fp), len(fn)


def run_evaluation(item,args,weights=None,export=True):
    # use topn most similar terms as words for aggregate freq if args.sims
    # get dataframe of evaluation metrics for each indivicual country
    k,words = item
    
    all_stats = get_country_stats(args.countries, words, 
                                  args.frequency_path,
                                  args.window, 
                                  args.months_prior, 
                                  args.method, 
                                  args.crisis_defs, 
                                  period=args.period,
                                  eval_end_date=args.eval_end_date,
                                  weights=weights,
                                  z_thresh=args.z_thresh)


    # Aggregate tp, fp, fn numbers for all countries to calc overall eval metrics
    tp, fp, fn = all_stats['tp'].sum(), all_stats['fp'].sum(), all_stats['fn'].sum()
    recall = get_recall(tp, fn)
    prec = get_precision(tp, fp)
    f2 = get_fscore(tp, fp, fn, beta=2)
    avg = pd.Series([recall, prec, f2, tp, fp, fn], 
                    name='aggregate', 
                    index=['recall','precision','fscore','tp','fp','fn'])
    all_stats = all_stats.append(avg)

    # Save to file and print results
    if export:
        all_stats.to_csv(os.path.join(args.eval_path,
                                      'agg_sim_{}_{}_offset_{}_smoothwindow_{}_{}_evaluation.csv'.format(args.sims,
                                                                                                       args.period,
                                                                                                       args.months_prior,
                                                                                                       args.window,
                                                                                                       k)))
    
    print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(k,words,recall, prec, f2))
    if args.weighted: 
        return k,list(zip(words,weights)),recall, prec, f2
    else:
        return k,words,recall, prec, f2
    #print('evaluated words: {}'.format(words))

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
    #%%
    # Get prec, rec, and fscore for each country for each word group
    #iter_items = list(search_words_sets.items())
    sks = list(search_words_sets.keys())
    #%%    
    z_thresh_range = np.arange(1,3.1,0.1)
    item = (sks[0],search_words_sets[sks[0]])
    args.countries = args.countries[:2]
    res_stats = run_evaluation(item,args,weights,False)
#%%
    print('new process')
    word_list = search_words_sets[sks[0]]
    r,p,f= eval_word_group(args.countries,word_list,
                                      period=args.period,
                                      stemmed=False,
                                      frequency_path=args.frequency_path,
                                      weights=weights,
                                      z_thresh=args.z_thresh)
##    
#    
#%%


#        ## export over all resoults to csv
#    df = pd.DataFrame(overall_res,columns=['word','sim_words','recall','prec','f2'])
#    df.to_csv(os.path.join(args.eval_path,'agg_sim_{}_overall_{}_offset_{}_smoothwindow_{}_evaluation.csv'.format(args.sims,args.period,args.months_prior,args.window)))
    

