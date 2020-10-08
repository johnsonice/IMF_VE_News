#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:28:33 2018

@author: chuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:48:38 2018

@author: chuang
"""

import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'./libs')
import config
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
from evaluate import get_recall,get_precision,get_fscore
from gensim.models.keyedvectors import KeyedVectors
import crisis_points
from mp_utils import Mp
from functools import partial as functools_partial


#%%
def get_stats(starts,ends,preds,offset,fbeta=2):
    tp, fn, mid_crisis  = [], [], []
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp() - offset, s.to_timestamp(), freq='q'), freq='q')
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp(), freq='q'), freq='q')
    
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
    
    return recall,precision,fscore

#def get_country_vocab(country,period='quarter',frequency_path=config.FREQUENCY):
#    data_path = os.path.join(frequency_path,'{}_{}_word_freqs.pkl'.format(country, period))
#    data = pd.read_pickle(data_path)
#    vocab = list(data.index)
#    return vocab

def get_sim_words(vecs,wg,topn):
    if not isinstance(wg,list): 
        wg = [wg]
    try:
        sims = [w[0] for w in vecs.wv.most_similar(wg, topn=topn)]
    except KeyError:
        try:
            wg_update = list()
            for w in wg:
                wg_update.extend(w.split('_'))
            sims = [w[0] for w in vecs.wv.most_similar(wg_update, topn=topn)]
            print('Warning: {} not in the vocabulary, split the word with _'.format(wg))
        except:
            print('Not in vocabulary: {}'.format(wg_update))
            return wg
    words = sims + wg
    return words


#%%
if __name__ == "__main__":
    period = config.COUNTRY_FREQ_PERIOD
    vecs = KeyedVectors.load(config.W2V)
    frequency_path = config.FREQUENCY
    countries = list(crisis_points.country_dict_original.keys())
    #countries = ['argentina']
    out_dir = '/data/News_data_raw/FT_WD_research/w2v_test/eval/time_series'

    def export_country_ts(country, period=period, vecs=vecs, frequency_path=frequency_path, out_dir=out_dir):
        series_wg = list()
        do_not_vectorize = ['positive_sentiment_language', 'negative_sentiment_language']
        for wg in config.targets:
            if wg in do_not_vectorize:
                word_groups =
            else:
                word_groups = get_sim_words(vecs,wg,15)
            df = aggregate_freq(word_groups, country,period=period,stemmed=False,frequency_path=frequency_path)
            df.name = wg
            series_wg.append(df)
        
        df_all = pd.concat(series_wg,axis=1)
        out_csv = os.path.join(out_dir, '{}_{}_time_series.csv'.format(country,period))
        df_all.to_csv(out_csv)
        
        return country, df_all

    if config.experimenting:
        class_type_setups = config.class_type_setups

        for setup in class_type_setups:
            class_type = setup[0]
            in_directory = os.path.join(frequency_path, class_type)
            out_directory = os.path.join(out_dir, class_type)
            if config.experiment_mode == "country_classification":
                export_country_ts_exp_1 = functools_partial(export_country_ts, frequency_path=in_directory,
                                                            out_dir=out_directory)
                mp = Mp(countries, export_country_ts_exp_1)
                res = mp.multi_process_files(chunk_size=1)

            elif config.experiment_mode == "topiccing_discrimination":
                for f2_thresh in config.topic_f2_thresholds:
                    if type(f2_thresh) is tuple:
                        f2_thresh = '{}_{}'.format(f2_thresh[0], f2_thresh[1])
                    else:
                        f2_thresh = str(f2_thresh)

                    for doc_thresh in config.document_topic_min_levels:
                        if type(doc_thresh) is tuple:
                            doc_thresh = '{}_{}'.format(doc_thresh[0], doc_thresh[1])
                        else:
                            doc_thresh = str(doc_thresh)

                        in_directory = os.path.join(config.topiccing_frequency, class_type, f2_thresh, doc_thresh)
                        out_directory = os.path.join(config.topiccing_eval_ts, class_type, f2_thresh, doc_thresh)

                        if config.just_five:
                            in_directory = os.path.join(in_directory, 'j5_countries')
                            out_directory = os.path.join(out_directory, 'j5_countries')

                        export_country_ts_exp_1 = functools_partial(export_country_ts, frequency_path=in_directory,
                                                                    out_dir=out_directory)
                        mp = Mp(countries, export_country_ts_exp_1)
                        res = mp.multi_process_files(chunk_size=1)

    else:

        mp = Mp(countries, export_country_ts)
        res = mp.multi_process_files(chunk_size=1)


def read_grouped_search_words(file_path):
    """
    Read the search words, by group, from the indicated file
    :param file_path:
    :return: 'dict' mapping key(group name) -> value(list of str words-in-group)
    """

    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k, v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]  # Save all non-NA values - different len groups
        temp_list = [wg.split('&') for wg in temp_list]   # split & for wv search
        search_groups[k] = temp_list  # Map str key group name -> list[str] group words
    return search_groups


def get_sim_words_set(args,word_group):
    """
    Return the top n most similar words based on the word2vec model

    :param args: configuration arguments
    :param word_group: list[str] of seed words
    :return: A 'set' of words associated with any of the passed words
    """

    assert isinstance(word_group,list)  # Must pass a list

    # Iterate over list
    sim_word_group = list()
    for w in word_group:

        # Save associated top n words, weights - if exist in the word2vec model
        try:
            words, weights = get_input_words_weights(args,
                                                 w,
                                                 vecs=vecs,
                                                 weighted=args.weighted)
            sim_word_group.extend(words)

        # If a word does not exist in the word2vec model, alert and ignore
        except:
            print('Not in vocabulary {}'.format(w))
    sim_word_set = set(sim_word_group)  # Eliminate duplicate associated words for group

    return sim_word_set


def run_evaluation(item,args,weights=None): # TODO clean up naming practice - wth is "item"
    """
    TODO docstring
    :param item:
    :param args:
    :param weights:
    :param export:
    :return:
    """
    # use topn most similar terms as words for aggregate freq if args.sims
    # get dataframe of evaluation metrics for each indivicual country
    k, words = item # TODO how is words defined

    # Get evaluation stats per country
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
    aggregate_tp, aggregate_fp, aggregate_fn = all_stats['tp'].sum(), all_stats['fp'].sum(), all_stats['fn'].sum()
    aggregate_recall = get_recall(aggregate_tp, aggregate_fn)
    aggregate_prec = get_precision(aggregate_tp, aggregate_fp)
    aggregate_f2 = get_fscore(aggregate_tp, aggregate_fp, aggregate_fn, beta=2)
    aggregate_stats = pd.Series([aggregate_recall, aggregate_prec, aggregate_f2, aggregate_tp, aggregate_fp, aggregate_fn],
                    name='aggregate',
                    index=['recall','precision','fscore','tp','fp','fn'])
    all_stats = all_stats.append(aggregate_stats)

    # Save to file if export
    if args.export:
        all_stats.to_csv(os.path.join(args.eval_path,
                                      'agg_sim_{}_{}_offset_{}_smoothwindow_{}_{}_evaluation.csv'.format(args.sims,
                                                                                                       args.period,
                                                                                                       args.months_prior,
                                                                                                       args.window,
                                                                                                       k)))

    # Print results if verbose
    if args.verbose:
        print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(k,words,aggregate_recall, aggregate_prec, aggregate_f2))
    if args.weighted:
        return k,list(zip(words,weights)),aggregate_recall, aggregate_prec, aggregate_f2
    else:
        return k,words,aggregate_recall, aggregate_prec, aggregate_f2
    #print('evaluated words: {}'.format(words))


def save_these_word_groups(all_groups, group_names):
    return_dict = {}
    for name in group_names:
        return_dict[name] = all_groups[name]

    return return_dict


def flatten_search_groups(groups_dict):
    re_dic = {}
    for key in groups_dict.keys():
        this_group = groups_dict[key]
        flat_group = [x[0] for x in this_group]
        re_dic[key] = flat_group

    return list(re_dic.items())


def get_group_items(search_groups):
    search_words_sets = dict()
    for k, v in search_groups.items():
        if args.sims:
            search_words_sets[k] = list(get_sim_words_set(args, search_groups[k]))  ## turn set to list
        else:
            search_words_sets[k] = [t for tl in v for t in tl]  ## flattern the list of list
    weights = None

    iter_items = list(search_words_sets.items())

    return iter_items


# TODO kick all the stuff out of above main and import it from 07_02, or move out of that one and create a utils file
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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

    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)

    args.verbose = True  # Todo modularize
    args.export = False  # TODO modularize
    args.eval_path = config.EXP_SEARCH_EVAL
    args.frequency_path = os.path.join(config.NEW_PROCESSING_FOLDER, 'frequency', 'csv', 'Min1_AllCountry') # TEMP

    group_search_file = os.path.join(config.SEARCH_TERMS, args.search_file)
    exp_files_directory = config.EXP_SEARCH_TERMS

    search_groups = read_grouped_search_words(group_search_file)

    def multi_run_eval(item, args=args, weights=None):
        # Get prec, rec, and fscore for each country for each word group
        res_stats = run_evaluation(item, args, weights)
        return res_stats

    # TEMP - non-perm TODO add based on flags or something?
    non_sentiment_names_list = ['fear_language', 'risk_language', 'hedging_language', 'opinion_language',
                                'crisis_language']
    sentiment_names_list = ['positive_sentiment_language', 'negative_sentiment_language']
    aggregate_names = ['agg_all_other_sentiments', 'agg_other_and_negative','all_language']
    base_names = non_sentiment_names_list + sentiment_names_list

    non_sentiment_groups = save_these_word_groups(search_groups, non_sentiment_names_list)
    sentiment_groups = save_these_word_groups(search_groups, sentiment_names_list)
    base_groups = save_these_word_groups(search_groups, base_names)

    non_sentiment_items = get_group_items(non_sentiment_groups)
    sentiment_items = flatten_search_groups(sentiment_groups)
    non_sentiment_items.extend(sentiment_items)
    base_items = get_group_items(base_groups)

    print("NON W2V items is :", non_sentiment_items)

    print("BASE ITEMS IS :", base_items)

    # run the evals

    mp_nonw2v = Mp(non_sentiment_items, multi_run_eval)
    nonw2v_res = mp_nonw2v.multi_process_files(workers=2,  # do not set workers to be too high, your memory will explode
                                               chunk_size=1)

    mp_w2v = Mp(base_items, multi_run_eval)
    w2v_res = mp_w2v.multi_process_files(workers=2,  # do not set workers to be too high, your memory will explode
                                         chunk_size=1)

    ## export over all results to csv
    df = pd.DataFrame(nonw2v_res, columns=['word', 'sim_words', 'recall', 'prec', 'f2'])
    save_file_full = os.path.join(args.eval_path,
                                  'group_words_without_w2v_on_sentiment.csv')
    df.to_csv(save_file_full)

    print("Saved at:", save_file_full)

    df = pd.DataFrame(w2v_res, columns=['word', 'sim_words', 'recall', 'prec', 'f2'])
    save_file_full = os.path.join(args.eval_path,
                                  'group_words_using_w2v_on_sentiment.csv')
    df.to_csv(save_file_full)

    print("Saved at:", save_file_full)

