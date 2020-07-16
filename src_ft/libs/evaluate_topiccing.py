"""
evaluate.py

description: collection of functions used in evaluation scripts
"""
import os
import numpy as np
import pandas as pd
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
from crisis_points import crisis_points,ll_crisis_points
from anomaly_detection_hpfilter_mad import anomaly_detection
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate_topic(tdf, country, topic,
             method='zscore',
             crisis_defs='kr', period='month',
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
    assert method in ('zscore', 'hpfilter')
    assert period in ('quarter', 'month', 'week', 'year', 'q', 'm', 'w', 'y')
    assert direction in ('incr', 'decr', None)
    assert crisis_defs in ('kr', 'll')
    fq = period[0].lower()
    assert fq in ('q', 'm')  ## make sure period is set to eight quarter or month

    if crisis_defs == 'kr':
        tdf = tdf[:pd.Period(eval_end_date[fq])] # Don't look beyond when Kaminsky and
        # Get start and 'end' periods for crises depending on definition
        starts = list(pd.PeriodIndex(crisis_points[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_points[country]['peaks'], freq=fq))
    elif crisis_defs == 'll':
        tdf = tdf[:pd.Period(eval_end_date[fq])] # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition
        starts = list(pd.PeriodIndex(ll_crisis_points[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(ll_crisis_points[country]['peaks'], freq=fq))

    ag_freq = tdf[str(topic)]
    ag_freq = ag_freq.apply(lambda x: float(x))

    preds = get_preds_from_pd(ag_freq,country,method, crisis_defs,period,
                             window, direction, months_prior, fbeta,
                             weights,z_thresh)
    if preds is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan  # return NAN if no mentions of topic words in country data

    # recall, precision, fscore, len(tp), len(fp), len(fn)
    return get_eval_stats(fq,starts,ends,preds,period,months_prior,fbeta)


def get_preds_from_pd(ag_freq,country,method='zscore', crisis_defs='kr',period='month',
                         window=24, direction='incr', months_prior=24, fbeta=2,
                         weights=None,z_thresh=1.96):
    # Value checks
    assert method in ('zscore','hpfilter')
    assert period in ('quarter', 'month', 'week', 'year', 'q', 'm', 'w', 'y')
    assert direction in ('incr', 'decr', None)
    assert crisis_defs in ('kr', 'll')
    fq = period[0].lower()
    assert fq in ('q','m')  ## make sure period is set to eight quarter or month

    ## sum frequency for specified words - it is pd series with time as index
    if not isinstance(ag_freq, pd.Series):
        print('\nno data for {}\n'.format(country))
        return None

    # Get periods for which desired method detects outliers
    if method == 'zscore':
        preds = list(signif_change(ag_freq,
                                   window,
                                   period=period,
                                   direction=direction,
                                   z_thresh=z_thresh).index)
        ## it return a list of time stamp e.g: [Period('2001Q3', 'Q-DEC')] or [Period('2001-03', 'M-DEC')]

    elif method == 'hpfilter':
        preds = anomaly_detection(ag_freq)

    return preds


def get_eval_stats(fq, starts, ends, preds, period, months_prior, fbeta=2):

    # Calc number of true positives, false positives and false negatives
    # True Positives: The number of anomalies that occured within t years of a crisis onset (i.e. within forecast window)
    # False positives: The number of anomalies occuring ouside of either crisis or forecast windows
    # False Negatives: The number of crises without an anomaly occuring in the forecast window
    offset = pd.DateOffset(months=months_prior)
    tp, fn, mid_crisis = [], [], []
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(how='s') - offset, s.to_timestamp(how='e'),freq=fq), freq=fq)
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(how='s'), e.to_timestamp(how='e'),freq=fq), freq=fq)

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

    #print(recall, precision, fscore, len(tp), len(fp), len(fn))
    return recall, precision, fscore, len(tp), len(fp), len(fn)


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


def get_fscore(tp, fp, fn, beta):
    try:
        return ((1+beta**2) * tp) / ((1+beta**2) * tp + (beta**2 * fn) + fp)
    except ZeroDivisionError:
        return np.nan


## get country specific statistics
def get_topic_stats(country, topic_list, read_folder, save_folder, window, months_prior, method,
                      crisis_defs, period, export=True, eval_end_date=None, weights=None, z_thresh=1.96):

    eval_periods = pd.date_range(start='1980-01', end='2002-01', freq='m').to_period('M')
    temp_df = pd.DataFrame(index=eval_periods)

    read_file = os.path.join(read_folder, '{}_100_topic_time_series.csv'.format(country))
    tdf = pd.read_csv(read_file)
    tdf = tdf.rename(columns={'Unnamed: 0': 'Date'})
    tdf = tdf.set_index('Date')
    tdf = tdf.join(temp_df, how='outer')

    topic_stats = []
    for topic in topic_list:

        stats = pd.Series(evaluate_topic(tdf, country, topic,
                                   window=window,
                                   months_prior=months_prior,
                                   method=method,
                                   period=period,
                                   crisis_defs=crisis_defs,
                                   eval_end_date=eval_end_date,
                                   weights=weights,
                                   z_thresh=z_thresh),
                          index=['recall','precision','fscore','tp','fp','fn'],
                          name=topic)  ## default period = quarter
        topic_stats.append(stats)
    all_topic = pd.DataFrame(topic_stats)

    # TEMP
    print("ALL TOPIC")
    print(all_topic)

    if export:
        all_topic.to_csv(os.path.join(save_folder, '{}_100_topic_eval.csv'.format(country)), index=False)

    return all_topic


## get w2v related words with weights
def get_input_words_weights(args,wg,vecs=None,weighted=False):
    # use topn most similar terms as words for aggregate freq if args.sims
    if args.sims:
        #vecs = KeyedVectors.load(args.wv_path)
        try:
            sims = [w for w in vecs.wv.most_similar(wg, topn=args.topn)]    ## get similar words and weights
        except KeyError:
            try:
                print('{} was splited for sim words searching..'.format(wg))
                wg_update = list()
                for w in wg:
                    wg_update.extend(w.split('_'))
                sims = [w for w in vecs.wv.most_similar(wg_update, topn=args.topn)]
            except:
                #print('Not in vocabulary: {}'.format(wg_update))
                raise Exception('Not in vocabulary: {}'.format(wg_update))

        wgw = [(w, 1) for w in wg]  ## assign weight 1 for original words
        words_weights = sims + wgw
    # otherwise the aggregate freq is just based on the term(s) in the current wg.
    else:
        wgw = [(w, 1) for w in wg]  ## assign weight 1 for original words
        words_weights = wgw

    ## get words and weights as seperate list
    words = [w[0] for w in words_weights]

    if weighted:
        weights = [w[1] for w in words_weights]
    else:
        weights = None

    return words, weights
