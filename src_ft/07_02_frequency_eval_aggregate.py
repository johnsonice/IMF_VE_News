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
    

class args_class(object):
    def __init__(self, targets,frequency_path=config.FREQUENCY,eval_path=config.EVAL_WG,
                 wv_path = config.W2V,topn=config.topn,months_prior=config.months_prior,
                 window=config.smooth_window_size,
                 countries=config.countries,
                 period=config.COUNTRY_FREQ_PERIOD,
                 eval_end_date=config.eval_end_date,
                 method='zscore',crisis_defs='kr',sims=True,weighted=False):
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
#%%
if __name__ == '__main__':
    
    ## load config arguments
    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,
                          sims=True,period=config.COUNTRY_FREQ_PERIOD, 
                          months_prior=config.months_prior,
                          window=config.smooth_window_size,
                          eval_end_date=config.eval_end_date,
                          weighted= config.WEIGHTED)

    # Parse input word groups, word_gropus is a list of list:
    # something like this: [['fear'],['worry'],['concern'],['risk'],['threat'],['warn'],['maybe']]
    file_path = os.path.join(config.SEARCH_TERMS,'grouped_search_words.csv')
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
    
    #%%
    # Get prec, rec, and fscore for each country for each word group
    overall_res = list()
    
    for k,words in search_words_sets.items(): 
        # use topn most similar terms as words for aggregate freq if args.sims
        
        # get dataframe of evaluation metrics for each indivicual country
        all_stats = get_country_stats(args.countries, words, 
                                      args.frequency_path,
                                      args.window, 
                                      args.months_prior, 
                                      args.method, 
                                      args.crisis_defs, 
                                      period=args.period,
                                      eval_end_date=args.eval_end_date,
                                      weights=weights)


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
                                      'agg_{}_offset_{}_smoothwindow_{}_{}_evaluation.csv'.format(args.period,
                                                                               args.months_prior,
                                                                               args.window,
                                                                               k)))

        #print('evaluated words: {}'.format(words))
        if args.weighted: 
            overall_res.append((k,list(zip(words,weights)),recall, prec, f2))
        else:
            overall_res.append((k,words,recall, prec, f2))
        print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(k,words,recall, prec, f2))

    ## export over all resoults to csv
    df = pd.DataFrame(overall_res,columns=['word','sim_words','recall','prec','f2'])
    df.to_csv(os.path.join(args.eval_path,'agg_overall_{}_offset_{}_smoothwindow_{}_evaluation.csv'.format(args.period,args.months_prior,args.window)))