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
from evaluate import evaluate, get_recall, get_precision, get_fscore 
import pandas as pd
import os
import config

def get_country_stats(countries, words, frequency_path, window, years_prior, method, crisis_defs,weights=None):
    country_stats = []
    for country in countries:
        stats = pd.Series(evaluate(words, country, frequency_path,window=window, 
                                   years_prior=years_prior, method=method, 
                                   crisis_defs=crisis_defs,weights=weights),
                          index=['recall','precision','fscore','tp','fp','fn'], 
                          name=country)  ## default period = quarter
        country_stats.append(stats)
    all_stats = pd.DataFrame(country_stats)
    return all_stats

def get_input_words_weights(args,wg,vecs=None):
    # use topn most similar terms as words for aggregate freq if args.sims
    if args.sims:
        #vecs = KeyedVectors.load(args.wv_path)
        try:
            sims = [w for w in vecs.wv.most_similar(wg, topn=args.topn)]                    ## get similar words and weights 
        except KeyError:
            try:
                print(wg)
                wg_update = list()
                for w in wg:
                    wg_update.extend(w.split('_'))
                sims = [w for w in vecs.wv.most_similar(wg_update, topn=args.topn)]
            except:
                #print('Not in vocabulary: {}'.format(wg_update))
                raise Exception('Not in vocabulary: {}'.format(wg_update))
                
        wgw = [(w,1) for w in wg]  ## assign weight 1 for original words
        words_weights = sims + wgw   
    # otherwise the aggregate freq is just based on the term(s) in the current wg.
    else:
        wgw = [(w,1) for w in wg]  ## assign weight 1 for original words
        words_weights = wgw
    
    ## get words and weights as seperate list
    words = [w[0] for w in words_weights]
    
    if args.weighted:    
        weights = [w[1] for w in words_weights]
    else:
        weights= None
    
    return words,weights
        
    
class args_class(object):
    def __init__(self, targets,frequency_path=config.FREQUENCY,eval_path=config.EVAL_WG,wv_path = config.W2V,topn=config.topn,years_prior=config.years_prior,
                 window=config.smooth_window_size,countries=config.countries,
                 method='zscore',crisis_defs='kr',sims=True,weighted=False):
        self.targets = targets
        self.frequency_path = frequency_path
        self.eval_path=eval_path
        self.wv_path = wv_path
        self.topn = topn
        self.years_prior = years_prior
        self.window = window
        self.countries = countries
        self.method = method
        self.crisis_defs = crisis_defs
        self.sims = sims
        self.weighted = weighted

if __name__ == '__main__':
    
    ## load config arguments
    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,
                          sims=True,weighted= config.WEIGHTED)

    # Parse input word groups
    word_groups = [wg.split('&') for wg in args.targets]
    
    if args.weighted:    
        print('Weighted flag = True ; Results are aggregated by weighted sum....')
        
    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)
    
    # Get prec, rec, and fscore for each country for each word group
    
    overall_res = list()
    
    for wg in word_groups: 
        # use topn most similar terms as words for aggregate freq if args.sims
        try:
            words, weights = get_input_words_weights(args,wg,vecs=vecs)
        except:
            print('Not in vocabulary: {}'.format(wg))
            continue
        
        # get dataframe of evaluation metrics for each indivicual country
        all_stats = get_country_stats(args.countries, words, args.frequency_path,args.window, 
                                      args.years_prior, args.method, args.crisis_defs,weights=weights)

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
        all_stats.to_csv(os.path.join(args.eval_path,'{}_evaluation.csv'.format('_'.join(wg))))

        #print('evaluated words: {}'.format(words))
        if args.weighted: 
            overall_res.append((wg,list(zip(words,weights)),recall, prec, f2))
        else:
            overall_res.append((wg,words,recall, prec, f2))
        print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(wg,words,recall, prec, f2))

    ## export over all resoults to csv
    df = pd.DataFrame(overall_res,columns=['word','sim_words','recall','prec','f2'])
    df.to_csv(os.path.join(args.eval_path,'overall_evaluation.csv'))