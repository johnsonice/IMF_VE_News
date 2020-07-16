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
from evaluate_topiccing import evaluate_topic, get_topic_stats
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

def eval_one_country(country, args,export=True):
    # use topn most similar terms as words for aggregate freq if args.sims
    # get dataframe of evaluation metrics for each indivicual country
    topics_list = range(args.num_topics)
    weighted = args.weighted
    read_folder = args.read_folder
    save_folder = args.save_folder

    all_topics = get_topic_stats(country, topics_list, read_folder, save_folder,
                                  args.frequency_path,
                                  args.window, 
                                  args.months_prior, 
                                  args.method, 
                                  args.crisis_defs,
                                  period=args.period,
                                 export=export,
                                  eval_end_date=args.eval_end_date,
                                    weights=weighted,
                                  z_thresh=args.z_thresh)


    # Aggregate tp, fp, fn numbers for all countries to calc overall eval metrics
    tp, fp, fn = all_topics['tp'].sum(), all_topics['fp'].sum(), all_topics['fn'].sum()
    recall = get_recall(tp, fn)
    prec = get_precision(tp, fp)
    f2 = get_fscore(tp, fp, fn, beta=2)
    avg = pd.Series([recall, prec, f2, tp, fp, fn], 
                    name='aggregate_topics_{}'.format(country),
                    index=['recall','precision','fscore','tp','fp','fn'])
    all_stats = all_topics.append(avg)

    # Save to file and print results
    if export:
        all_stats.to_csv(os.path.join(save_folder,
                                      'agg_{}_100_topic_evaluation.csv'.format(country)))
    
    print('Evaluated country {}'.format(country))

    return avg
    #print('evaluated words: {}'.format(words))


def eval_countries(args, export=True):
    countries = args.countries
    overall_stats = []
    for country in countries:
        country_avgs = eval_one_country(country, args, export)
        overall_stats.append(country_avgs)

    overall_df = pd.DataFrame(overall_stats)

    print("OVERALL DF")
    print(overall_df)

    if export:
        overall_df.to_csv(os.path.join(args.save_folder,'overall_100_topic_evaluation.csv'))

    return overall_df

#        
#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('-t', '--targets', action='store', dest='targets', default=config.targets)
    parser.add_argument('-f', '--frequency_path', action='store', dest='frequency_path', default=config.FREQUENCY)
    parser.add_argument('-c', '--countries', action='store', dest='countries', default=config.countries)
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

    args.read_folder = '/data/News_data_raw/FT_WD_research/topiccing/time_series/Min1_AllCountry'
    args.save_folder = '/data/News_data_raw/FT_WD_research/topiccing/eval/Min1_AllCountry'

    args.num_topics = 100
    args.weighted = False
    args.period = 'm'

    class_type_setups = config.class_type_setups
    eval_type = config.eval_type
    original_eval_path = args.eval_path

    for setup in class_type_setups:

        eval_countries(args)

