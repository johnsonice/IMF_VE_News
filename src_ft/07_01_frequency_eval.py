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

def get_country_stats(countries, words, frequency_path, window, months_prior, method, 
                      crisis_defs,period,eval_end_date=None,weights=None):
    country_stats = []
    for country in countries:
        stats = pd.Series(evaluate(words, 
                                   country, 
                                   frequency_path,
                                   window=window, 
                                   months_prior=months_prior, 
                                   method=method, 
                                   period=period,
                                   crisis_defs=crisis_defs,
                                   eval_end_date=eval_end_date,
                                   weights=weights),
                          index=['recall','precision','fscore','tp','fp','fn'], 
                          name=country)  ## default period = quarter
        country_stats.append(stats)
    all_stats = pd.DataFrame(country_stats)
    return all_stats

def get_input_words_weights(args,wg,vecs=None,weighted=False):
    # use topn most similar terms as words for aggregate freq if args.sims
    if args.sims:
        #vecs = KeyedVectors.load(args.wv_path)
        try:
            sims = [w for w in vecs.wv.most_similar(wg, topn=args.topn)]    ## get similar words and weights 
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
    
    if weighted:    
        weights = [w[1] for w in words_weights]
    else:
        weights= None
    
    return words,weights
        
    
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
    word_groups = [wg.split('&') for wg in args.targets]
    
    if args.weighted:    
        print('Weighted flag = True ; Results are aggregated by weighted sum....')
        
    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)
    
    # Get prec, rec, and fscore for each country for each word group
    
    overall_res = list()
    
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
            if isinstance(wg,list):
                words = wg 
            else:
                words = [wg]
        
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
                                      '{}_offset_{}_smoothwindow_{}_{}_evaluation.csv'.format(args.period,
                                                                               args.months_prior,
                                                                               args.window,
                                                                               '_'.join(wg))))

        #print('evaluated words: {}'.format(words))
        if args.weighted: 
            overall_res.append((wg,list(zip(words,weights)),recall, prec, f2))
        else:
            overall_res.append((wg,words,recall, prec, f2))
        print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(wg,words,recall, prec, f2))

    ## export over all resoults to csv
    df = pd.DataFrame(overall_res,columns=['word','sim_words','recall','prec','f2'])
    df.to_csv(os.path.join(args.eval_path,'overall_{}_offset_{}_smoothwindow_{}_evaluation.csv'.format(args.period,args.months_prior,args.window)))