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
from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore 
import pandas as pd
import os
import config

def get_country_stats(countries, words, frequency_path, window, years_prior, method, crisis_defs):
    country_stats = []
    for country in countries:
        stats = pd.Series(evaluate(words, country, frequency_path,window=window, 
                                   years_prior=years_prior, method=method, 
                                   crisis_defs=crisis_defs),
                          index=['recall','precision','fscore','tp','fp','fn'], 
                          name=country)  ## default period = quarter
        country_stats.append(stats)
    all_stats = pd.DataFrame(country_stats)
    return all_stats

class args_class(object):
    def __init__(self, targets,frequency_path=config.FREQUENCY,eval_path=config.EVAL_WG,wv_path = config.W2V,topn=config.topn,years_prior=config.years_prior,
                 window=config.smooth_window_size,countries=config.countries,
                 method='zscore',crisis_defs='kr',sims=True):
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


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('targets', nargs='+', 
                            help="""list of terms to check. space delimits term groups, and & 
                            delimits terms within groups. 
                            e.g. 'fear worry&risk' is ['fear'], ['worry','risk']""")
        parser.add_argument('-n', '--topn', action='store', dest='topn', 
                            default=15, type=int)
        parser.add_argument('-yp', '--years_prior', action='store', dest='years_prior',
                            default=2, type=int)
        parser.add_argument('-w', '--window', action='store', dest='window', 
                            default=8, type=int)
        parser.add_argument('-c', '--countries', action='store', dest='countries', 
                            default=crisis_points.keys())
        parser.add_argument('-m', '--method', action='store', dest='method', 
                            default='zscore')
        parser.add_argument('-cd', '--crisis_defs', action='store', dest='crisis_defs', 
                            default='kr')
        parser.add_argument('-sims', '--sims', action='store', dest='sims', type=bool, 
                            default=True)
        parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', 
                            default='../models/vsms/word_vecs_5_20_200')
        
        args = parser.parse_args()
    except:
        args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,sims=True)

    # Parse input word groups
    word_groups = [wg.split('&') for wg in args.targets]
    
    # Get prec, rec, and fscore for each country for each word group
    for wg in word_groups: 
        # use topn most similar terms as words for aggregate freq if args.sims
        if args.sims:
            vecs = KeyedVectors.load(args.wv_path)
            try:
                sims = [w[0] for w in vecs.wv.most_similar(wg, topn=args.topn)]
            except KeyError:
                try:
                    print(wg)
                    wg_update = list()
                    for w in wg:
                        wg_update.extend(w.split('_'))
                    sims = [w[0] for w in vecs.wv.most_similar(wg_update, topn=args.topn)]
                except:
                    print('Not in vocabulary: {}'.format(wg_update))
                    continue
            words = sims + wg
        # otherwise the aggregate freq is just based on the term(s) in the current wg.
        else:
            words = wg
        
        
        # get dataframe of evaluation metrics for each indivicual country
        all_stats = get_country_stats(args.countries, words, args.frequency_path,args.window, 
                                      args.years_prior, args.method, args.crisis_defs)

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
        print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(wg,words,recall, prec, f2))

        