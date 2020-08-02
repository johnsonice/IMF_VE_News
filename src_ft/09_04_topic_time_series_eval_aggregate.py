"""
frequency_eval.py

Description: Used to evaluate supplied terms and term groups wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval.py <TERM1> <TERM2> ...
NOTE: to see an explanation of optional arguments, use python3 frequency_eval.py --help
"""
import sys
sys.path.insert(0, './libs')
import argparse
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
from evaluate_topiccing import evaluate_topic, get_topic_stats
import pandas as pd
import os
import config


def eval_one_country(country, args,export=True):
    # use topn most similar terms as words for aggregate freq if args.sims
    # get dataframe of evaluation metrics for each individual country
    topics_list = range(args.num_topics)
    weighted = args.weighted
    read_folder = args.read_folder
    save_folder = args.save_folder

    # Weight monthly topic values by a factor of the in-month document count
    if weighted:
        raise NotImplementedError("The weighted mode of topic evaluation is unsupported at present")

    # Get all per-topic results for this country (df)
    all_topics = get_topic_stats(country, topics_list, read_folder, save_folder, args.window, args.months_prior,
                                 args.method, args.crisis_defs, period=args.period, export=export,
                                 eval_end_date=args.eval_end_date, weights=weighted, z_thresh=args.z_thresh)

    print('Evaluated country {}'.format(country))

    # Save to file and print confirmation
    if export:
        export_file_name = os.path.join(save_folder, 'agg_{}_{}_period_{}_topic_evaluation.csv'.format(
            country, config.COUNTRY_FREQ_PERIOD, config.num_topics))
        all_topics.to_csv(export_file_name)

        print('Topic evaluation saved at {}'.format(export_file_name))

    # Also returns the df in-memory
    return all_topics


def eval_countries(args, export=True):
    """
    Evaluate all of the countries, one by one
    :param args:
    :param export:
    :return:
    """
    countries = args.countries
    for country in countries:
        country_avgs = eval_one_country(country, args, export)


def eval_aggregate_topics(args, export=True):
    """
    Assess the predictive ability of all
    :param args:
    :param export:
    :return:
    """
    countries = args.countries
    save_folder = args.save_folder
    num_topics = args.num_topics

    # Create dataframe filled with 0s
    agg_df = pd.DataFrame(index=range(num_topics), columns=['recall', 'precision', 'f2-score', 'tp', 'fp', 'fn'],
                          data=[[0] * 6] * num_topics)

    # Look through all the countries
    for country in countries:

        # Read country topic evaluation
        read_file_name = os.path.join(save_folder, 'agg_{}_{}_period_{}_topic_evaluation.csv'.format(
            country, config.COUNTRY_FREQ_PERIOD, config.num_topics))
        tdf = pd.read_csv(read_file_name)

        # Take sums of predictions across all the topics
        for top_num in range(num_topics):
            agg_df.loc[top_num, 'tp'] = agg_df.loc[top_num, 'tp'] + tdf.loc[top_num, 'tp']
            agg_df.loc[top_num, 'fp'] = agg_df.loc[top_num, 'fp'] + tdf.loc[top_num, 'fp']
            agg_df.loc[top_num, 'fn'] = agg_df.loc[top_num, 'fn'] + tdf.loc[top_num, 'fn']

    # Calculate and save recall, precision, f2 for each topic
    for top_num in range(num_topics):
        tp = agg_df.loc[top_num, 'tp']
        fp = agg_df.loc[top_num, 'fp']
        fn = agg_df.loc[top_num, 'fn']
        recall = get_recall(tp, fn)
        prec = get_precision(tp, fp)
        f2 = get_fscore(tp, fp, fn, beta=2)
        agg_df.loc[top_num, 'recall'] = recall
        agg_df.loc[top_num, 'precision'] = prec
        agg_df.loc[top_num, 'f2-score'] = f2

    print("Cross-country topic time series evaluated")

    # Save to file if exporting
    if export:
        save_name = os.path.join(save_folder, 'cross_country_{}_period_{}_topic_eval.csv'.format(
            num_topics, config.COUNTRY_FREQ_PERIOD))
        agg_df.to_csv(save_name)
        print('Cross-country topic evaluation saved ad {}'.format(save_name))

    # Also return df in-memory
    return agg_df


if __name__ == '__main__':

    # Read arguments from bash
    # TODO delete uncessary
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
    #parser.add_argument('-p', '--period', action='store', dest='period', default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-mp', '--months_prior', action='store', dest='months_prior', default=config.months_prior)
    parser.add_argument('-w', '--window', action='store', dest='window',default=config.smooth_window_size)
    parser.add_argument('-eed', '--eval_end_date', action='store', dest='eval_end_date',default=config.eval_end_date)
    parser.add_argument('-wed', '--weighted', action='store_true', dest='weighted',default=config.WEIGHTED)
    parser.add_argument('-z', '--z_thresh', action='store', dest='z_thresh',type=int, default=config.z_thresh)
    parser.add_argument('-gsf', '--search_file', action='store', dest='search_file',default=config.GROUPED_SEARCH_FILE)
    args = parser.parse_args()

    # TODO clean up
    args.read_folder = os.path.join(config.topiccing_time_series,'Min1_AllCountry')
    args.save_folder = os.path.join(config.topiccing_eval_levels_ts, 'Min1_AllCountry')

    args.num_topics = config.num_topics
    args.weighted = config.topiccing_level_weighted
    args.period = config.COUNTRY_FREQ_PERIOD

    class_type_setups = config.class_type_setups
    eval_type = config.eval_type
    original_eval_path = args.eval_path
    # To here


    # Iterate over all the setups - evaluate the within-country topic power
    for setup in class_type_setups:
        args.setup_name = setup[0]
        eval_countries(args)

    # Iterate over all the setups - evaluate the aggregate topics power
    for setup in class_type_setups:
        args.setup_name = setup[0]
        eval_aggregate_topics(args)
