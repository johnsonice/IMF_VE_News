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
import pandas as pd
import numpy as np
from mp_utils import Mp
import os
import config



def get_key_sim_pair(word_groups,args,vecs):
    key_sim_pairs = []
    for wg in word_groups:
        if args.sims:
            # use topn most similar terms as words for aggregate freq if args.sims
            try:
                # get words and weights. weights will be 1s if weight flag is false
                # otherwise weights will be cos distance 
                words, weights = get_input_words_weights(args,
                                                         wg,
                                                         vecs=vecs,
                                                         weighted=args.weighted)
            except:
                print('Not in vocabulary: {}'.format(wg))
                continue
        else:
            weights= None  ## if not using w2v , set weights to None
            if isinstance(wg,list):
                words = wg 
            else:
                words = [wg]
        
        key_sim_pairs.append((wg,words,weights))
    
    return key_sim_pairs

def run_evaluation(iter_item,args):  
    ## unpack iter items 
    wg,words,weights = iter_item
    # get dataframe of evaluation metrics for each indivicual country
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
    all_stats.to_csv(os.path.join(args.eval_path,
                                  '{}_offset_{}_smoothwindow_{}_{}_evaluation.csv'.format(args.period,
                                                                           args.months_prior,
                                                                           args.window,
                                                                           '_'.join(wg))))
    
    print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(wg,words,recall, prec, f2))
    
    if args.weighted: 
        return wg,list(zip(words,weights)),recall, prec, f2
    else:
        return wg,words,recall, prec, f2
    
    
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
        self.z_thresh=z_thresh

if __name__ == '__main__':
    
    ## load config arguments
    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,
                          sims=config.SIM,period=config.COUNTRY_FREQ_PERIOD, 
                          months_prior=config.months_prior,
                          window=config.smooth_window_size,
                          eval_end_date=config.eval_end_date,
                          weighted= config.WEIGHTED,
                          z_thresh = config.z_thresh)

    # Parse input word groups, word_gropus is a list of list:
    # something like this: [['fear'],['worry'],['concern'],['risk'],['threat'],['warn'],['maybe']]
    word_groups = [wg.split('&') for wg in args.targets]
    
    if args.weighted:    
        print('Weighted flag = True ; Results are aggregated by weighted sum....')
        
    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)
    
    # Get prec, rec, and fscore for each country for each word group
    iter_items = get_key_sim_pair(word_groups,args,vecs)

    #overall_res = list()
    # run in multi process
    def multi_run_eval(wg,args=args):
        res_stats = run_evaluation(wg,args)
        return res_stats
    
    mp = Mp(iter_items,multi_run_eval)
    overall_res = mp.multi_process_files(workers=5, chunk_size=1)  ## do not set workers to be too high, your memory will explode
    
    ## export over all resoults to csv
    df = pd.DataFrame(overall_res,columns=['word','sim_words','recall','prec','f2'])
    df.to_csv(os.path.join(args.eval_path,'overall_{}_offset_{}_smoothwindow_{} _evaluation.csv'.format(args.period,args.months_prior,args.window)))
