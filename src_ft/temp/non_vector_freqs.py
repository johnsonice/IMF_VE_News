import sys, os
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import pandas as pd
import config
import crisis_points
from frequency_utils import rolling_z_score, aggregate_freq, signif_change, all_word_count
from mp_utils import Mp



# %%
if __name__ == "__main__":
    period = config.COUNTRY_FREQ_PERIOD
    #vecs = KeyedVectors.load(config.W2V)
    frequency_path = '/data/News_data_raw/FT_WD_research/frequency/temp/All_Comb'
    countries = list(crisis_points.country_dict_original.keys())
    # countries = ['argentina']
    out_dir = '/data/News_data_raw/FT_WD_research/w2v_test/eval/time_series'

    #words_to_lists_df = pd.read_csv('/home/apsurek/IMF_VE_News/research/w2v_compare/words_mapping.csv')
    words_to_lists_df = pd.read_csv('/home/apsurek/IMF_VE_News/research/w2v_compare/words_mapping_3.csv')

    positive_targs = ['able', 'enable', 'adequately', 'benign', 'buoyant', 'buoyancy', 'comfortable', 'confident', 'enhance', 'favorable', 'favourably', 'healthy', 'improve', 'improvement', 'mitigate', 'positive', 'positively', 'profits', 'rebound', 'recover', 'recovery', 'resilience', 'resilient', 'solid', 'sound', 'stabilise', 'stabilize', 'success', 'successful', 'successfully']
    negative_targs = ['abrupt', 'adverse', 'adversely', 'aggravate', 'bad', 'burden', 'challenge', 'closure', 'contraction', 'costly', 'damage', 'danger', 'deficit', 'dent', 'destabilise', 'deteriorate', 'deterioration', 'deterioration', 'difficult', 'discourage', 'downgrade', 'drag', 'erode', 'erosion', 'exacerbate', 'expose', 'fear', 'force', 'fragility', 'gloomy', 'hurt', 'illiquid', 'impairment', 'inability', 'jeopardise', 'lose', 'negative', 'pose', 'question', 'repercussion', 'risky', 'severely', 'shortfall', 'spiral', 'squeeze', 'stagnate', 'strain', 'stress', 'struggle', 'suffer', 'threaten', 'turbulent', 'unable', 'undermine', 'unease', 'unexpectedly', 'vulnerable', 'weakness', 'worsen', 'writedowns']
    non_vec_targs = list(set(positive_targs + negative_targs)) # Should be irrelevant

    negation_targ = ['negation']

    def fetch_list(word):
        return list(words_to_lists_df[word].dropna())

    def export_country_ts(country, period=period, frequency_path=frequency_path, out_dir=out_dir):
        series_wg = list()
        for wg in non_vec_targs:
            # df = aggregate_freq([wg], country, period=period, stemmed=False, frequency_path=frequency_path)
            read_list = fetch_list(wg)
            df = aggregate_freq(read_list, country, period=period, stemmed=False, frequency_path=frequency_path)
            df.name = wg
            series_wg.append(df)

        df_all = pd.concat(series_wg, axis=1)
        #out_csv = os.path.join(out_dir, '{}_{}_time_series_non_vec.csv'.format(country, period))
        #out_csv = os.path.join(out_dir, '{}_{}_time_series_cherry_picked.csv'.format(country, period))
        out_csv = os.path.join(out_dir, '{}_{}_time_series_cherry_picked_3.csv'.format(country, period))
        df_all.to_csv(out_csv)

        return country, df_all

    def export_country_negative_count(country, period=period, frequency_path=frequency_path, out_dir=out_dir):
        series_wg = list()
        negation_list = ['aint', 'cannot', "can't", "daren't", "didn't", "doesn't", "don't", "hadn't", 'hardly',
                         "hasn't", "haven't", "havn't", "isn't", 'lack', 'lacking', 'lacks', 'neither', 'never',
                         'no', 'nobody', 'none', 'nor', 'not', 'nothing', 'nowhere', "mightn't", "mustn't", "needn't",
                         "oughtn't", "shan't", "shouldn't", "wasn't", "without", "wouldn't"]

        df = aggregate_freq(negation_list, country, period=period, stemmed=False, frequency_path=frequency_path)
        df.name = 'negation_index'
        series_wg.append(df)

        df_all = pd.concat(series_wg, axis=1)
        #out_csv = os.path.join(out_dir, '{}_{}_time_series_non_vec.csv'.format(country, period))
        #out_csv = os.path.join(out_dir, '{}_{}_time_series_cherry_picked.csv'.format(country, period))
        out_csv = os.path.join(out_dir, '{}_{}_negation.csv'.format(country, period))
        df_all.to_csv(out_csv)

        return country, df_all

    def get_word_counts(country, period=period, frequency_path=frequency_path, out_dir=out_dir):
        series_wg = list()
        df = all_word_count(country, period=period, stemmed=False, frequency_path=frequency_path)
        df.name = 'total_word_count'
        series_wg.append(df)

        df_all = pd.concat(series_wg, axis=1)
        #out_csv = os.path.join(out_dir, '{}_{}_time_series_non_vec.csv'.format(country, period))
        #out_csv = os.path.join(out_dir, '{}_{}_time_series_cherry_picked.csv'.format(country, period))
        out_csv = os.path.join(out_dir, '{}_{}_word_counts_fix_2.csv'.format(country, period))
        df_all.to_csv(out_csv)

        return country, df_all

    for country in countries:
        #export_country_negative_count(country)
        get_word_counts(country)
