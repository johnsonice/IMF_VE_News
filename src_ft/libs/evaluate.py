"""
evaluate.py

description: collection of functions used in evaluation scripts
"""
import numpy as np
import pandas as pd
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
from crisis_points import crisis_points
from anomaly_detection_hpfilter_mad import anomaly_detection
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate(word_list, country, frequency_path,method='zscore', crisis_defs='kr', period='quarter', 
             stemmed=False, window=8, direction='incr', years_prior=2, fbeta=2):
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
    assert method in ('zscore','hpfilter')
    assert period in ('quarter','month','week','year')
    assert direction in ('incr', 'decr', None)
    assert crisis_defs in ('kr', 'fund')

    # Setup
    ag_freq = aggregate_freq(word_list, country, period, stemmed,frequency_path) ## sum frequency for specified words - it is pd series with time as index
    if not isinstance(ag_freq, pd.Series):
        print('\nno data for {}\n'.format(country))
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan # return NAN if no mentions of topic words in country data
    offset = pd.DateOffset(years=years_prior)

    # Get start and 'end' periods for crises depending on definition
    if crisis_defs == 'kr':
        ag_freq = ag_freq[:'1999Q2'] # Don't look beyond when Kaminsky and Reinhart puyblished in 1999
        starts = list(pd.PeriodIndex(crisis_points[country]['starts'], freq='q'))
        ends = list(pd.PeriodIndex(crisis_points[country]['peaks'], freq='q'))
    elif crisis_defs == 'fund':
        crises = pd.read('../data/crises.csv')
        country_crises = crises[crises['country_name'] == country]['years']
        starts = [pd.Period('{}-01'.format(year), freq='q') for year in set(country_crises)]
        ends = [pd.Period('{}-01'.format(int(year) + 1), freq='q') for year in set(country_crises)]

    # Get periods for which desired method detects outliers
    if method == 'zscore':
        preds = list(signif_change(ag_freq, window, direction).index) ## it return a list of time stamp e.g: [Period('2001Q3', 'Q-DEC')]
    elif method == 'hpfilter':
        preds = anomaly_detection(ag_freq)

    # Calc number of true positives, false positives and false negatives
    # True Positives: The number of anomalies that occured within t years of a crisis onset (i.e. within forecast window)
    # False positives: The number of anomalies occuring ouside of either crisis or forecast windows
    # False Negatives: The number of crises without an anomaly occuring in the forecast window
    tp, fn, mid_crisis  = [], [], []
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp() - offset, s.to_timestamp()), freq='q')
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp()), freq='q')

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
    except Zero :
        return np.nan
