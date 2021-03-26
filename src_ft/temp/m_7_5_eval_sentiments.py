"""
frequency_eval.py

Description: Used to evaluate supplied terms and term groups wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval.py <TERM1> <TERM2> ...
NOTE: to see an explanation of optional arguments, use python3 frequency_eval.py --help
"""
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
#import argparse
from gensim.models.keyedvectors import KeyedVectors
from crisis_points import crisis_points_TEMP_KnR, crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats, \
    get_preds_from_pd, get_eval_stats
import pandas as pd
import numpy as np
from mp_utils import Mp
import os
import config


def get_key_sim_pair(word_groups,args,vecs):
    key_sim_pairs = []
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
            weights= None  ## if not using w2v , set weights to None
            if isinstance(wg,list):
                words = wg
            else:
                words = [wg]

        key_sim_pairs.append((wg,words,weights))

    return key_sim_pairs

def run_evaluation(iter_item,args):
    ## unpack iter items
    wg,words,weights = iter_item
    # get dataframe of evaluation metrics for each indivicual country
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

    print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(wg,words,recall, prec, f2))

    if args.weighted:
        return wg,list(zip(words,weights)),recall, prec, f2
    else:
        return wg,words,recall, prec, f2


class args_class(object):
    def __init__(self, targets=config.targets,frequency_path=config.FREQUENCY,eval_path=config.EVAL_WG,
                 wv_path = config.W2V,topn=config.topn,months_prior=config.months_prior,
                 window=config.smooth_window_size,
                 countries=config.countries,
                 period=config.COUNTRY_FREQ_PERIOD,
                 eval_end_date=config.eval_end_date,
                 method='zscore',crisis_defs='kr',
                 sims=True,weighted=False,z_thresh=config.z_thresh):
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
        self.z_thresh=z_thresh


def evaluate(frequency_ser, country, method='zscore',
             crisis_defs='kr', period='month', stemmed=False,
             window=24, direction='incr', months_prior=24, fbeta=2,
             eval_end_date=None, weights=None, z_thresh=1.96):
    """
    evaluates how the aggregate frequency of the provided word list performs based on the evaluation method
    and the crisis definitions provided.

    params:
        word_list: (list) list of words the aggregate freq of which you want to evalutate
        country: (str) country name
        method: (str) evaluation method
        crisis_defs: (str) which crisis time frames to use
        period: (str) time granularity over which to eval aggregate freq
        stemmed: (bool) whether to use stemmed token counts or not (this arg gets passed to the streamer eventually)
        window: (int) number of periods over which moving average of aggregate freq and z-score will be calc'ed
        direction: (str or nonetype) ['decr', 'incr', None] which significant differences to count as hits in the
            z-score eval. if 'incr', only significant increases will be counted as hits. if 'decr', only significant
            decreases. if None, both.
        years_prior: (int) this determines the forecast window. e.g. if set to 2, any hits within 2 years of a crisis
            onset will be considered a true positive.
        fbeta: (int) beta value to use in fscore calculation. 2 is default, since recall is more important for this task.
            NB beta determines how you weight recall wrt precision. 2 means recall is weighted as twice as important.
    """
    # Value checks
    '''
    assert method in ('zscore', 'hpfilter')
    assert period in ('quarter', 'month', 'week', 'year')
    assert direction in ('incr', 'decr', None)
    assert crisis_defs in ('kr', 'll')'''  # Chill again
    fq = period[0].lower()
    assert fq in ('q', 'm')  ## make sure period is set to eight quarter or month

    # Setup

    if type(frequency_ser) != pd.core.series.Series:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if crisis_defs == 'kr':
        #eval_end_date = pd.to_datetime(eval_end_date)
        #fq = pd.to_datetime(fq, format='%Y-%m')
        ag_freq = frequency_ser#[:eval_end_date[fq]]  # Don't look beyond when Kaminsky and
        # Get start and 'end' periods for crises depending on definition
        starts = list(pd.PeriodIndex(crisis_points_TEMP_KnR[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_points_TEMP_KnR[country]['peaks'], freq=fq))

    elif crisis_defs == 'll':
        ag_freq = frequency_ser[:eval_end_date[fq]]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition
        starts = list(pd.PeriodIndex(ll_crisis_points[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(ll_crisis_points[country]['peaks'], freq=fq))

    elif crisis_defs == 'IMF_GAP_6':
        end = '2019-12'
        ag_freq = frequency_ser[:end]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition
        crisis_dict = crisis_points.imf_gap_6_events
        starts = list(pd.PeriodIndex(crisis_dict[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_dict[country]['peaks'], freq=fq))

    elif crisis_defs == 'IMF_GAP_0':
        end = '2019-12'
        ag_freq = frequency_ser[:end]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition

        crisis_dict = crisis_points.imf_all_events
        starts = list(pd.PeriodIndex(crisis_dict[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_dict[country]['peaks'], freq=fq))

    elif crisis_defs == 'RomerRomer':
        end = '2012-12'
        ag_freq = frequency_ser[:end]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition

        crisis_dict = crisis_points.crisis_points_RomerNRomer
        starts = list(pd.PeriodIndex(crisis_dict[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_dict[country]['peaks'], freq=fq))

    elif crisis_defs == 'LoDuca':
        end = '2016-12'
        ag_freq = frequency_ser[:end]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition

        crisis_dict = crisis_points.crisis_points_LoDuca
        starts = list(pd.PeriodIndex(crisis_dict[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_dict[country]['peaks'], freq=fq))

    elif crisis_defs == 'ReinhartRogoff':
        end = '2014-12'
        ag_freq = frequency_ser[:end]

        crisis_dict = crisis_points.crisis_points_Reinhart_Rogoff_All
        starts = list(pd.PeriodIndex(crisis_dict[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_dict[country]['peaks'], freq=fq))

    elif crisis_defs in ['IMF_Monthly_Starts', 'IMF_Monthly_Starts_Gap_3', 'IMF_Monthly_Starts_Gap_6']:
        assess_dict = {
            'IMF_Monthly_Starts': crisis_points.imf_programs_monthly,
            'IMF_Monthly_Starts_Gap_3': crisis_points.imf_programs_monthly_gap3,
            'IMF_Monthly_Starts_Gap_6': crisis_points.imf_programs_monthly_gap6
        }

        end = '2019-12'
        ag_freq = frequency_ser[:end]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition
        crisis_dict = assess_dict[crisis_defs]
        starts = list(pd.PeriodIndex(crisis_dict[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_dict[country]['peaks'], freq=fq))

    else:
        raise ValueError("Wrong crisis_defs value presented")

    preds = get_preds_from_pd(ag_freq, country, method, crisis_defs, period,
                              window, direction, months_prior, fbeta,
                              weights, z_thresh)
    if preds is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan  # return NAN if no mentions of topic words in country data

    # recall, precision, fscore, len(tp), len(fp), len(fn)
    return get_eval_stats(fq, starts, ends, preds, period, months_prior, fbeta)

def get_recall(tp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return np.nan


def get_precision(tp, fp):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return np.nan


def get_fscore(tp, fp, fn, beta=2):
    try:
        return ((1+beta**2) * tp) / ((1+beta**2) * tp + (beta**2 * fn) + fp)
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':

    ## load config arguments
    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
                          countries = config.countries,wv_path = config.W2V,
                          sims=config.SIM,period=config.COUNTRY_FREQ_PERIOD,
                          months_prior=config.months_prior,
                          window=config.smooth_window_size,
                          eval_end_date=config.eval_end_date,
                          weighted= config.WEIGHTED,
                          z_thresh = config.z_thresh)


    in_directory = '/data/News_data_raw/FT_WD_research/eval/word_defs/series'
    in_name = os.path.join(in_directory, '{}_sentiment_indeces.csv')
    out_name = os.path.join('/data/News_data_raw/FT_WD_research/eval/word_defs/series', '{}_sentiment_eval.csv')

    df = pd.read_csv(in_name.format('argentina'))
    sent_cols = df.columns[4:].values
    overall_tp, overall_fp, overall_fn = np.zeros(shape=len(sent_cols)), np.zeros(shape=len(sent_cols)), \
                                         np.zeros(shape=len(sent_cols))

    overall_df = pd.DataFrame({'sentiment':sent_cols,'tp':overall_tp, 'fp':overall_fp, 'fn':overall_fn})
    overall_df = overall_df.set_index('sentiment')

    config.countries = ['argentina']

    for ctry in config.countries:
        in_f = in_name.format(ctry)
        df = pd.read_csv(in_f)
        sent_cols = np.append(df.columns[3:8].values, df.columns[10:12].values)
        sent_cols = np.append(sent_cols, ['vader_pos_x_fed_pos', 'vader_neg_x_fed_neg',
       'vader_is_pos_x_fed_pos', 'vader_is_neg_x_fed_neg'])
        df['month'] = pd.to_datetime(df['month'])
        df.set_index('month')
        recls, precs, fscrs, ntps, nfps, nfns = [], [], [], [], [], []
        for sent_def in sent_cols:

            freq_ser = df[sent_def]
            recall, precision, fscore, ntp, nfp, nfn = evaluate(freq_ser, ctry,method='zscore',crisis_defs='kr',
                                                          period=args.period,stemmed=False,
                                                          window=args.window, direction='incr',
                                                                months_prior=args.months_prior,
                                                          fbeta=2,eval_end_date=args.eval_end_date,weights=None,
                                                                z_thresh=args.z_thresh)

            recls.append(recall)
            precs.append(precision)
            fscrs.append(fscore)
            ntps.append(ntp)
            nfps.append(nfp)
            nfns.append(nfns)

            overall_df.loc[sent_def, 'tp'] += ntp
            overall_df.loc[sent_def, 'fp'] += nfp
            overall_df.loc[sent_def, 'fn'] += nfn

        df_out = pd.DataFrame({
            'recall':recls,
            'precision':precs,
            'fscore':fscrs,
            'tp': ntps,
            'fp': nfps,
            'fn': nfns
        })

        df_out['sentiment_def'] = sent_cols

        df_out.to_csv(out_name.format(ctry))

    orec = []
    opre = []
    of2 = []
    for sent_def in sent_cols:
        tp = overall_df.loc[sent_def, 'tp']
        fp = overall_df.loc[sent_def, 'fp']
        fn = overall_df.loc[sent_def, 'fn']
        orec.append(get_recall(tp,fn))
        opre.append(get_precision(tp,fp))
        of2.append(get_fscore(tp,fp,fn))

    overall_df['recall'] = orec
    overall_df['precision'] = opre
    overall_df['fscore'] = of2
    overall_df.sort_values(by='fscore', ascending=False)
    overall_df.to_csv('/home/apsurek/IMF_VE_News/research/overall_sentiment_analysis.csv')
