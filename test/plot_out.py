import sys
sys.path.insert(0,'./libs')
sys.path.insert(0,'..')
from plot_utils import plot_frequency

import argparse
from gensim.models.keyedvectors import KeyedVectors
from evaluate import evaluate, get_recall, get_precision, get_fscore, get_input_words_weights, get_country_stats
import pandas as pd
import os
from mp_utils import Mp
import config


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
        temp_list = [wg.split('&') for wg in temp_list]  # split & for wv search
        search_groups[k] = temp_list  # Map str key group name -> list[str] group words
    return search_groups


def get_sim_words_set(args, word_group):
    """
    Return the top n most similar words based on the word2vec model

    :param args: configuration arguments
    :param word_group: list[str] of seed words
    :return: A 'set' of words associated with any of the passed words
    """

    assert isinstance(word_group, list)  # Must pass a list

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


def run_evaluation(item, args, weights=None):  # TODO clean up naming practice - wth is "item"
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
    k, words = item  # TODO how is words defined

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
    aggregate_stats = pd.Series(
        [aggregate_recall, aggregate_prec, aggregate_f2, aggregate_tp, aggregate_fp, aggregate_fn],
        name='aggregate',
        index=['recall', 'precision', 'fscore', 'tp', 'fp', 'fn'])
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
        print(
            '\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(k, words, aggregate_recall,
                                                                                            aggregate_prec,
                                                                                            aggregate_f2))
    if args.weighted:
        return k, list(zip(words, weights)), aggregate_recall, aggregate_prec, aggregate_f2
    else:
        return k, words, aggregate_recall, aggregate_prec, aggregate_f2
    # print('evaluated words: {}'.format(words))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--targets', action='store', dest='targets', default=config.targets)
    parser.add_argument('-f', '--frequency_path', action='store', dest='frequency_path', default=config.FREQUENCY)
    parser.add_argument('-c', '--countries', action='store', dest='countries', default=config.countries)
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', default=config.W2V)
    parser.add_argument('-ep', '--eval_path', action='store', dest='eval_path', default=config.EVAL_WG)
    parser.add_argument('-md', '--method', action='store', dest='method', default='zscore')
    parser.add_argument('-cd', '--crisis_defs', action='store', dest='crisis_defs', default=config.crisis_defs)
    parser.add_argument('-sims', '--sims', action='store', dest='sims', default=config.SIM)
    parser.add_argument('-tn', '--topn', action='store', dest='topn', type=int, default=config.topn)
    parser.add_argument('-p', '--period', action='store', dest='period', default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-mp', '--months_prior', action='store', dest='months_prior', default=config.months_prior)
    parser.add_argument('-w', '--window', action='store', dest='window', default=config.smooth_window_size)
    parser.add_argument('-eed', '--eval_end_date', action='store', dest='eval_end_date', default=config.eval_end_date)
    parser.add_argument('-wed', '--weighted', action='store_true', dest='weighted', default=config.WEIGHTED)
    parser.add_argument('-z', '--z_thresh', action='store', dest='z_thresh', type=int, default=config.z_thresh)
    parser.add_argument('-gsf', '--search_file', action='store', dest='search_file', default=config.GROUPED_SEARCH_FILE)
    args = parser.parse_args()

    args.verbose = True  # Todo modularize
    args.export = True  # TODO modularize

    # Parse input word groups, word_gropus is a list of list:
    # something like this: [['fear'],['worry'],['concern'],['risk'],['threat'],['warn'],['maybe']]

    file_path = os.path.join(config.SEARCH_TERMS, args.search_file)
    search_groups = read_grouped_search_words(file_path)

    ## it is a dictionary list:
    #       {'fear_language': [['fear']],
    #       'risk_language': [['threat'], ['warn']]}

    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)

    if args.weighted:
        raise Exception('for now, this module only works for unweighted calculation')
        # print('Weighted flag = True ; Results are aggregated by weighted sum....')
    else:
        search_words_sets = dict()
        for k, v in search_groups.items():
            if args.sims:
                search_words_sets[k] = list(get_sim_words_set(args, search_groups[k]))  ## turn set to list
            else:
                search_words_sets[k] = [t for tl in v for t in tl]  ## flattern the list of list
        weights = None

    iter_items = list(search_words_sets.items())

    class_type = 'Min1_AllCountry'
    countries = ['argentina']
    period = config.COUNTRY_FREQ_PERIOD

    directories_to_pull_from = [
        "/data/News_data_raw/FT_WD_research/frequency/csv/Min1_AllCountry",  # base comparison

        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.3/0.01/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.3/0.05/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.3/0.25/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.3/top_10/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.3/top_20/j5_countries",

        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.4/0.01/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.4/0.05/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.4/0.25/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.4/top_10/j5_countries",
        "/data/News_data_raw/FT_WD_research/topiccing/frequency/Min1_AllCountry/0.4/top_20/j5_countries",
    ]

    ex_pull = directories_to_pull_from[0]
    data_path_csv = os.path.join(ex_pull, '{}_{}_word_freqs.csv'.format(countries[0], period))
    data = pd.read_csv(data_path_csv, index_col=0)
    word_group = iter_items[0][0]

    ex_graph = plot_frequency(data, words=word_group, country=countries[0])

    ex_write_out_fold = '/home/apsurek/IMF_VE_News/test'
    out_file = os.path.join(ex_write_out_fold, "sample_argentina_graph.jpg")

    ex_graph.savefig(out_file)
    print('Saved figure at', out_file)


