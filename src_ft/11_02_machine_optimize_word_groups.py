import sys
sys.path.insert(0, './libs')
import argparse
from gensim.models.keyedvectors import KeyedVectors
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
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
    groups = read_grouped_search_words(args.search_file)

    args.verbose = True  # Todo modularize
    args.export = True  # TODO modularize

    if args.sims:
        vecs = KeyedVectors.load(args.wv_path)

    read_file_path = os.path.join(config.SEARCH_TERMS, args.search_file)
    write_file_path = os.path.join(config.EXP_SEARCH_TERMS, args.search_file)

    search_groups = read_grouped_search_words(read_file_path)

    search_words_sets = dict()
    for k, v in search_groups.items():
        if args.sims:
            search_words_sets[k] = list(get_sim_words_set(args, search_groups[k])) ## turn set to list
        else:
            search_words_sets[k] = [t for tl in v for t in tl] ## flattern the list of list
    weights = None

    iter_items = list(search_words_sets.items())

    def multi_optimize_word_groups(search_group, args=args, weights=None):
        base_item = get_sim_words_set(args, search_group)
        base_eval = run_evaluation(item, args, weights)


    def multi_optimize_word_groups(item, args=args, weights=None):
        # Get prec, rec, and fscore for each country for each word group


        return res_stats


    def main_process(args, iter_items):
        '''
        The meat of the program - evaluates the time series provided to it, saves the result to a properly named file
            and directory
        :param args:
        :param iter_items:
        :return:
        '''

        # run the eval function in multi process mode
        mp = Mp(iter_items, multi_run_eval)
        overall_res = mp.multi_process_files(workers=2,  # do not set workers to be too high, your memory will explode
                                             chunk_size=1)

        ## export over all resoults to csv
        df = pd.DataFrame(overall_res, columns=['word', 'sim_words', 'recall', 'prec', 'f2'])
        save_file_full = os.path.join(args.eval_path,
                                      'overall_agg_sim_{}_overall_{}_offset_{}_smoothwindow_{}_evaluation.csv'.format(
                                          args.sims, args.period, args.months_prior, args.window))
        df.to_csv(save_file_full)
        print("Saved at:", save_file_full)

    # If experimenting
    class_type_setups = config.class_type_setups
    eval_type = config.eval_type
    original_eval_path = args.eval_path
    original_freq_path = args.frequency_path

    # iterate over different country-document classification
    for setup in class_type_setups:
        class_type = setup[0]

        freq_path = os.path.join(original_freq_path, class_type)  # Moved the TF_DFs manually for speed since 06_0
        ev_path = os.path.join(original_eval_path, class_type)

        args.frequency_path = freq_path
        args.eval_path = ev_path

        # Execute the process setups times
        main_process(args, iter_items)
