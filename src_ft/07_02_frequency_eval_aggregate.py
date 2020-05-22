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
import argparse
from gensim.models.keyedvectors import KeyedVectors
#from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
import pandas as pd
#import numpy as np
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
      
#        
#%%
if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-t', '--targets', action='store', dest='targets', default=config.targets)
    parser.add_argument('-f', '--frequency_path', action='store', dest='frequency_path', default=config.FREQUENCY)
    parser.add_argument('-c', '--countries', action='store', dest='countries', default=['argentina'])
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', default=config.W2V)
    parser.add_argument('-ep', '--eval_path', action='store', dest='eval_path', default=config.EVAL_WG)
    parser.add_argument('-md', '--method', action='store', dest='method', default='zscore')
    parser.add_argument('-cd', '--crisis_defs', action='store', dest='crisis_defs', default=config.crisis_defs)
    parser.add_argument('-sims', '--sims', action='store', dest='sims', default=config.SIM)
    parser.add_argument('-tn', '--topn', action='store', dest='topn',type=int, default=config.topn)    
    parser.add_argument('-p', '--period', action='store', dest='period', default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-mp', '--months_prior', action='store', dest='months_prior', default=config.months_prior)
    parser.add_argument('-w', '--window', action='store', dest='window',default=config.smooth_window_size)
    parser.add_argument('-eed', '--eval_end_date', action='store', dest='eval_end_date',default=config.eval_end_date)
    parser.add_argument('-wed', '--weighted', action='store_true', dest='weighted',default=config.WEIGHTED)
    parser.add_argument('-z', '--z_thresh', action='store', dest='z_thresh',type=int, default=config.z_thresh)
    parser.add_argument('-gsf', '--search_file', action='store', dest='search_file',default=config.GROUPED_SEARCH_FILE)
    args = parser.parse_args()

    # Parse input word groups, word_gropus is a list of list:
    # something like this: [['fear'],['worry'],['concern'],['risk'],['threat'],['warn'],['maybe']]
 
    file_path = os.path.join(config.SEARCH_TERMS,args.search_file)
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
    iter_items = list(search_words_sets.items())

    # run in multi process
    def multi_run_eval(item,args=args,weights=None):
        res_stats = run_evaluation(item,args,weights)
        return res_stats
    
    mp = Mp(iter_items,multi_run_eval)
    overall_res = mp.multi_process_files(workers=5, chunk_size=1)  ## do not set workers to be too high, your memory will explode
    
        ## export over all resoults to csv
    df = pd.DataFrame(overall_res,columns=['word','sim_words','recall','prec','f2'])
    df.to_csv(os.path.join(args.eval_path,'overall_agg_sim_{}_overall_{}_offset_{}_smoothwindow_{}_evaluation.csv'.format(args.sims,args.period,args.months_prior,args.window)))
#    

