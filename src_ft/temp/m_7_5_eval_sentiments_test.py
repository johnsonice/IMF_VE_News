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
import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats, \
    get_preds_from_pd, get_eval_stats
import pandas as pd
import numpy as np
import os
import config

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

    #print('\n\n{}:\nevaluated words: {}\n\trecall: {}, precision: {}, f-score: {}'.format(wg,words,recall, prec, f2))

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

    #print('Evaluating on :', frequency_ser)
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
        ag_freq = frequency_ser[:eval_end_date[fq]]  # Don't look beyond when Kaminsky and
        # Get start and 'end' periods for crises depending on definition
        starts = list(pd.PeriodIndex(crisis_points.crisis_points_TEMP_KnR[country]['starts'], freq=fq))
        #print('Starts are:', starts)
        ends = list(pd.PeriodIndex(crisis_points.crisis_points_TEMP_KnR[country]['peaks'], freq=fq))
        #print('Ends are:', ends)

    elif crisis_defs == 'll':
        ag_freq = frequency_ser[:eval_end_date[fq]]  # Don't look beyond when ll ends
        # Get start and 'end' periods for crises depending on definition
        starts = list(pd.PeriodIndex(crisis_points.ll_crisis_points[country]['starts'], freq=fq))
        ends = list(pd.PeriodIndex(crisis_points.ll_crisis_points[country]['peaks'], freq=fq))

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

def get_countries(crisis_def):
    # Return the countries associated with each crisis definition
    if crisis_def == 'kr':
        #return crisis_points.crisis_points_TEMP_KnR.keys()
        return ['argentina']
    if crisis_def == 'll':
        return crisis_points.ll_crisis_points.keys()
    if crisis_def == 'IMF_GAP_6':
        return crisis_points.imf_gap_6_events.keys()
    if crisis_def == 'IMF_GAP_0':
        return crisis_points.imf_all_events.keys()
    if crisis_def == 'RomerRomer':
        return crisis_points.crisis_points_RomerNRomer.keys()
    if crisis_def == 'LoDuca':
        return crisis_points.crisis_points_LoDuca.keys()
    if crisis_def == 'ReinhartRogoff':
        return crisis_points.crisis_points_Reinhart_Rogoff_All.keys()
    if crisis_def == 'IMF_Monthly_Starts':
        return crisis_points.imf_programs_monthly.keys()
    if crisis_def == 'IMF_Monthly_Starts_Gap_3':
        return crisis_points.imf_programs_monthly_gap3.keys()
    if crisis_def == 'IMF_Monthly_Starts_Gap_6':
        return crisis_points.imf_programs_monthly_gap6.keys()


def create_agg_index(index_words, all_word_freq):
    agg_index = pd.Series(name=index_words.name, index=all_word_freq.index)
    index_word_vals = list(index_words.dropna().values)
    print("$$$ {} : TYPE {}".format(index_word_vals, type(index_word_vals)))

    for ind in all_word_freq.index:
        # Read one-by-one words in case no values
        this_val = 0
        for word in index_word_vals:
            try:
                this_val += all_word_freq[word].loc[ind]
            except:
                this_val += 0
        #print('Ind {} this val {}'.format(ind, this_val))
        agg_index.loc[ind] = this_val

    return agg_index

def get_compare_frame(desired_indeces, sims_map, all_word_freq, idx):
    compare_frame = pd.DataFrame(index=all_word_freq.index)
    compare_frame.index.name = 'month'
    for col in desired_indeces:
        agg_index = create_agg_index(sims_map[col].dropna(), all_word_freq)
        #print(agg_index)
        compare_frame = compare_frame.join(agg_index)

    compare_frame = compare_frame.reindex(idx, fill_value=0)
    return compare_frame

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

    #in_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_mean2')
    #in_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_mean_new_test')
    in_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_mean_new_test_sum')

    #in_name = os.path.join(in_dir, '{}_month_sentiment_indeces.csv')
    in_name = os.path.join(in_dir, '{}_month_sentiment.csv')
    out_name = os.path.join(config.EVAL_WordDefs,'indecy_eval_test_sum_compare', '{}_sentiment_eval_on_{}_crisis_def.csv')

    df_a = pd.read_csv(in_name.format('argentina'))
    sent_cols = df_a.columns[1:7]

    df_a['month'] = pd.to_datetime(df_a['month'])
    df_a = df_a.set_index('month')
    idx = df_a.index

    sims_map = pd.read_csv('../libs/all_sims_maps.csv') # works?
    compare_dir = '/data/News_data_raw/FT_WD_research/frequency/temp/All_Comb'
    compare_freq_base = os.path.join(compare_dir, '{}_month_word_freqs.csv') # Have to generate the compare in based on the freq data
    compare_out = os.path.join(config.EVAL_WordDefs,'indecy_eval_test_sum_compare', '{}_sentiment_eval_on_{}_crisis_def_COMPARE.csv')
    desired_indeces = df_a.columns[0:6]
    #desired_indeces = df_a.columns[0:1]

    # Temp
    compare_freq_f = compare_freq_base.format('argentina')
    #compare_out = compare_out.format('argentina')
    compare_freq = pd.read_csv(compare_freq_f).set_index('Unnamed: 0').T.fillna(value=0)
    compare_frame = get_compare_frame(desired_indeces, sims_map, compare_freq, idx)
    compare_frame.to_csv(os.path.join(config.EVAL_WordDefs,'indecy_eval_test_sum_compare','compare_frame.csv'))

    # Match inputs from the paper (?)
    args.window = 24
    args.months_prior = 24
    #args.months_prior = 18
    args.z_thresh = 2
    #args.z_thresh = 1.96

    #crisis_definitions = ['kr', 'll', 'IMF_GAP_6', 'IMF_GAP_0', 'RomerRomer', 'LoDuca',
    #               'ReinhartRogoff', 'IMF_Monthly_Starts', 'IMF_Monthly_Starts_Gap_3',
    #               'IMF_Monthly_Starts_Gap_6']
    crisis_definitions = ['kr']

    # all_sentiment_frame uses multi-index on crisis_def, sentiment_def
    midx = pd.MultiIndex.from_product([crisis_definitions,sent_cols])
    all_sentiment_frame = pd.DataFrame(index=midx,columns=['recall','precision','f2score','tp','fp','fn'])

    no_data_countries = []

    for crisis_def in crisis_definitions:
        print('Working on crisis def', crisis_def)
        #countries = config.countries
        #countries = config.countries # TODO SWAP ^ add other crisis defs
        countries = get_countries(crisis_def)

        overall_tp, overall_fp, overall_fn = np.zeros(shape=len(sent_cols)), np.zeros(shape=len(sent_cols)), \
                                             np.zeros(shape=len(sent_cols))

        overall_df = pd.DataFrame({'sentiment': sent_cols, 'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn})
        overall_df = overall_df.set_index('sentiment')

        for ctry in countries:
            print('Working on ', ctry)
            in_f = in_name.format(ctry)
            try:
                df = pd.read_csv(in_f)
            except:
                print('Cannot read', ctry)
                no_data_countries.append(ctry)
                continue
            if df.empty:
                print('File but no data for', ctry)
                no_data_countries.append(ctry)
                continue
            #df = df[df['country'] == ctry]
            df['month'] = pd.to_datetime(df['month'])
            df = df.set_index('month')

            #re-index for missing dates
            df = df.reindex(idx, fill_value=0)

            recls, precs, fscrs, ntps, nfps, nfns = [], [], [], [], [], []
            for sent_def in sent_cols:
                print('\tWorking on', sent_def)
                freq_ser = df[sent_def]
                recall, precision, fscore, ntp, nfp, nfn = evaluate(freq_ser, ctry, method='zscore', crisis_defs=crisis_def,
                                                              period=args.period,stemmed=False,
                                                              window=args.window, direction='incr',
                                                                    months_prior=args.months_prior,
                                                              fbeta=2,eval_end_date=args.eval_end_date, weights=None,
                                                                    z_thresh=args.z_thresh)

                recls.append(recall)
                precs.append(precision)
                fscrs.append(fscore)
                ntps.append(ntp)
                nfps.append(nfp)
                nfns.append(nfn)

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

            df_out_name = out_name.format(ctry, crisis_def)
            df_out.to_csv(df_out_name)
            print('Saved {} at {}'.format(ctry, df_out_name))

        ###$$$$$ TEMP
            recls, precs, fscrs, ntps, nfps, nfns = [], [], [], [], [], []
            for sent_def in desired_indeces:
                print('\tWorking on', sent_def)
                freq_ser = compare_frame[sent_def]
                recall, precision, fscore, ntp, nfp, nfn = evaluate(freq_ser, ctry, method='zscore', crisis_defs=crisis_def,
                                                              period=args.period,stemmed=False,
                                                              window=args.window, direction='incr',
                                                                    months_prior=args.months_prior,
                                                              fbeta=2,eval_end_date=args.eval_end_date, weights=None,
                                                                    z_thresh=args.z_thresh)

                recls.append(recall)
                precs.append(precision)
                fscrs.append(fscore)
                ntps.append(ntp)
                nfps.append(nfp)
                nfns.append(nfn)

            df_out = pd.DataFrame({
                'recall':recls,
                'precision':precs,
                'fscore':fscrs,
                'tp': ntps,
                'fp': nfps,
                'fn': nfns
            })

            df_out['sentiment_def'] = desired_indeces

            df_out_name = compare_out.format(ctry, crisis_def)
            df_out.to_csv(df_out_name)
            print('COMPARE Saved {} at {}'.format(ctry, df_out_name))
        ###$$$$$

        # Save and print overall predictive quality on this crisis defintion
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
        #overall_out_name = os.path.join(config.EVAL_WordDefs,'indecy_eval', f'all_country_overall_sentiment_{crisis_def}_eval.csv')
        overall_out_name = os.path.join(config.EVAL_WordDefs,'indecy_eval_test', f'all_country_overall_sentiment_{crisis_def}_eval.csv')

        overall_df.to_csv(overall_out_name)
        print(f'Saved overall stats for relevant countries, {crisis_def} crisis definitions saved at {overall_out_name}')

        # Save to overall crisis defs for comparison
        #all_sentiment_frame.loc[crisis_def][:] = overall_df

    #all_sent_name = os.path.join(config.EVAL_WordDefs,'indecy_eval', f'all_country_all_defs_overall_sentiment_eval.csv')
    #all_sentiment_frame.to_csv(all_sent_name)
    #print(f'Saved overall stats for relevant countries, all definitions saved at {all_sent_name}')
    print('\n\nCountries with no data:\n', no_data_countries)

