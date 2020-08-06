import sys
sys.path.insert(0, './libs')
import argparse
from gensim.models.keyedvectors import KeyedVectors
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
import pandas as pd
import os
from mp_utils import Mp
import config
from 07_02_frequency_eval_aggregate import read_grouped_search_words, get_sim_words_set, run_evaluation

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
    args.eval_path = config.EXP_SEARCH_EVAL
    args.frequency_path = os.path.join(config.NEW_PROCESSING_FOLDER, 'frequency', 'csv', 'Min1_AllCountry') # TEMP

    group_search_file = os.path.join(config.SEARCH_TERMS, args.search_file)
    exp_files_directory = config.EXP_SEARCH_TERMS

    search_groups = read_grouped_search_words(group_search_file)

    def save_these_word_groups(all_groups, group_names):
        return_dict = {}
        for name in group_names:
            return_dict[name] = all_groups[name]

        return return_dict

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

    def multi_run_eval(item, args=args, weights=None):
        # Get prec, rec, and fscore for each country for each word group
        res_stats = run_evaluation(item, args, weights)
        return res_stats

    # TEMP - non-perm TODO add based on flags or something?
    non_sentiment_names_list = ['fear_language','risk_language', 'hedging_language', 'opinion_language',
                                'crisis_language']
    sentiment_names_list = ['positive_sentiment_language', 'negative_sentiment_language']
    aggregate_names = ['agg_all_other_sentiments','agg_other_and_negative','all_language']
    base_names = non_sentiment_names_list + sentiment_names_list

    non_sentiment_groups = save_these_word_groups(search_groups, non_sentiment_names_list)
    sentiment_groups = save_these_word_groups(search_groups, sentiment_names_list)
    base_groups = save_these_word_groups(search_groups, base_names)

    non_sentiment_items = get_group_items(non_sentiment_groups)
    sentiment_items = sentiment_groups
    base_items = get_group_items(base_groups)

    name_item_dict = {'non_sentiment':}

    # run the evals

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

