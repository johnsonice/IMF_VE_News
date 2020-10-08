import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import pandas as pd
import config

# %%
def get_stats(starts, ends, preds, offset, fbeta=2):
    tp, fn, mid_crisis = [], [], []
    for s, e in zip(starts, ends):
        forecast_window = pd.PeriodIndex(pd.date_range(s.to_timestamp() - offset, s.to_timestamp(), freq='q'), freq='q')
        crisis_window = pd.PeriodIndex(pd.date_range(s.to_timestamp(), e.to_timestamp(), freq='q'), freq='q')

        period_tp = []
        # Collect True positives and preds happening during crisis
        for p in preds:
            if p in forecast_window:  # True Positive if prediction occurs in forecast window
                period_tp.append(p)
            elif p in crisis_window:  # if pred happened during crisis, don't count as fp
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

    return recall, precision, fscore


# def get_country_vocab(country,period='quarter',frequency_path=config.FREQUENCY):
#    data_path = os.path.join(frequency_path,'{}_{}_word_freqs.pkl'.format(country, period))
#    data = pd.read_pickle(data_path)
#    vocab = list(data.index)
#    return vocab

def get_sim_words(vecs, wg, topn):
    if not isinstance(wg, list):
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


# %%
if __name__ == "__main__":
    period = config.COUNTRY_FREQ_PERIOD
    #vecs = KeyedVectors.load(config.W2V)
    frequency_path = config.FREQUENCY
    countries = list(crisis_points.country_dict_original.keys())
    # countries = ['argentina']
    out_dir = '/data/News_data_raw/FT_WD_research/w2v_test/eval/time_series'

    positive_targs = ['able', 'enable', 'adequately', 'benign', 'buoyant', 'buoyancy', 'comfortable', 'confident', 'enhance', 'favorable', 'favourably', 'healthy', 'improve', 'improvement', 'mitigate', 'positive', 'positively', 'profits', 'rebound', 'recover', 'recovery', 'resilience', 'resilient', 'solid', 'sound', 'stabilise', 'stabilize', 'success', 'successful', 'successfully']
    negative_targs = ['abrupt', 'adverse', 'adversely', 'aggravate', 'bad', 'burden', 'challenge', 'closure', 'contraction', 'costly', 'damage', 'danger', 'deficit', 'dent', 'destabilise', 'deteriorate', 'deterioration', 'deterioration', 'difficult', 'discourage', 'downgrade', 'drag', 'erode', 'erosion', 'exacerbate', 'expose', 'fear', 'force', 'fragility', 'gloomy', 'hurt', 'illiquid', 'impairment', 'inability', 'jeopardise', 'lose', 'negative', 'pose', 'question', 'repercussion', 'risky', 'severely', 'shortfall', 'spiral', 'squeeze', 'stagnate', 'strain', 'stress', 'struggle', 'suffer', 'threaten', 'turbulent', 'unable', 'undermine', 'unease', 'unexpectedly', 'vulnerable', 'weakness', 'worsen', 'writedowns']
    non_vec_targs = list(set(positive_targs + negative_targs)) # Should be irrelevant

    def export_country_ts(country, period=period, vecs=vecs, frequency_path=frequency_path, out_dir=out_dir):
        series_wg = list()
        for wg in non_vec_targs:
            df = aggregate_freq(wg, country, period=period, stemmed=False, frequency_path=frequency_path)
            df.name = wg
            series_wg.append(df)

        df_all = pd.concat(series_wg, axis=1)
        out_csv = os.path.join(out_dir, '{}_{}_time_series_non_vec.csv'.format(country, period))
        df_all.to_csv(out_csv)

        return country, df_all


    mp = Mp(countries, export_country_ts)
    res = mp.multi_process_files(chunk_size=1)