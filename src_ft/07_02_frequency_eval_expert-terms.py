"""
frequency_eval_expert-terms.py

Description: Used to evaluate supplied terms in a supplied csv term list wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval_expert-terms.py <TERM_LIST_PATH>
NOTE: to see an explanation of optional arguments, use python3 frequency_eval_expert-terms.py --help
"""
import sys
sys.path.insert(0,'./libs')
import argparse
from gensim.models.keyedvectors import KeyedVectors
from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore 
from frequency_eval import get_country_stats
import pandas as pd
import os

class args_class(object):
    def __init__(self, term_list,frequency_path='../data/frequency',eval_path='../data/eval/experts',
                 topn=15,years_prior=2,window=8,countries=crisis_points.keys(),
                 method='zscore',crisis_defs='kr',wv=True):
        self.term_list = term_list
        self.frequency_path = frequency_path
        self.eval_path=eval_path
        self.topn = topn
        self.years_prior = years_prior
        self.window = window
        self.countries = countries
        self.method = method
        self.crisis_defs = crisis_defs
        self.wv = wv


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('term_list', help='path to term list (csv file)') 
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
        parser.add_argument('-cd', '--frequency_path', action='store', dest='frequency_path', 
                            default='kr')
        parser.add_argument('-freqp', '--crisis_defs', action='store', dest='crisis_defs', 
                            default='../data/frequency')
        parser.add_argument('-evalp', '--eval_path', action='store', dest='eval_path', 
                            default='../data/eval/experts')
        parser.add_argument('-wv', '--word_vectors', action='store', dest='wv', default=True,
                            type=bool, help='True for get topn most similar, False for \
                            just the words themselves')
        args = parser.parse_args()
    except:
        args = args_class('../data/search_terms/experts/crisis_expert_terms.csv',wv=False)

    # Parse supplied terms
    term_list = pd.read_csv(args.term_list)
    terms = [i[0].lower() for i in term_list.values]

    if args.wv:
        vecs = KeyedVectors.load('../models/vsms/word_vecs_5_20_200')
    
    # Get prec, rec, and fscore for each country for each wg
    term_data = pd.DataFrame([], index=terms, columns=['recall','precision','fscore','tp','fp','fn'])
    for i, w in enumerate(terms): 
        print('\rWorking on {} of {} terms'.format(i, len(terms)), end='')
        if args.wv:
            try:
                words = [t[0] for t in vecs.most_similar(w, topn=args.topn)]
                words.append(w)
            except KeyError:
                continue
        else:
            words = [w]

        # get dataframe of evaluation metrics for each indivicual country
        all_stats = get_country_stats(args.countries, words, args.frequency_path,args.window, 
                                      args.years_prior, args.method, args.crisis_defs)
        # Aggregate tp, fp, fn numbers for all countries to calc overall eval metrics
        all_tp, all_fp, all_fn = all_stats['tp'].sum(), all_stats['fp'].sum(), all_stats['fn'].sum()
        all_recall = get_recall(all_tp, all_fn)
        all_prec = get_precision(all_tp, all_fp)
        all_f2 = get_fscore(all_tp, all_fp, all_fn, beta=2)
        term_stats = pd.Series([all_recall, all_prec, all_f2, all_tp, all_fp, all_fn], 
                               name=w, 
                               index=['recall','precision','fscore','tp','fp','fn'])
        term_data.loc[w] = term_stats 
        all_stats = all_stats.append(term_stats)
        all_stats.to_csv(os.path.join(args.eval_path,'{}_evaluation.csv'.format(w)))

    # Save results
    vec_flag = '' if args.wv else '_no_semantic_cluster'
    term_data.to_csv(os.path.join(args.eval_path,'expert_list_evaluation{}.csv'.format(vec_flag)))







