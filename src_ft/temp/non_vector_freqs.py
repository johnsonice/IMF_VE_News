import sys, os
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import pandas as pd
import config
import crisis_points
from frequency_utils import rolling_z_score, aggregate_freq, signif_change
from mp_utils import Mp



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

    def export_country_ts(country, period=period, frequency_path=frequency_path, out_dir=out_dir):
        series_wg = list()
        for wg in non_vec_targs:
            df = aggregate_freq([wg], country, period=period, stemmed=False, frequency_path=frequency_path)
            df.name = wg
            series_wg.append(df)

        df_all = pd.concat(series_wg, axis=1)
        out_csv = os.path.join(out_dir, '{}_{}_time_series_non_vec.csv'.format(country, period))
        df_all.to_csv(out_csv)

        return country, df_all

    for country in countries:
        export_country_ts(country)
