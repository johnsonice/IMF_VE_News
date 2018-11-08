from stream import SentStreamer_fast as SentStreamer
from stream import FileStreamer_fast as FileStreamer
import os
from nltk.corpus import stopwords
from region_mapping import region
import pandas as pd
from crisis_points import crisis_points
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
from collections import defaultdict
#from plot_utils import plot_frequency


def aggregate_freq(word_list, country,period='quarter', stemmed=False,frequency_path='../data/frequency'):
    assert isinstance(word_list, list), 'Must pass a list to aggregate_freq'
    s_flag = '_stemmed' if stemmed else ''
    #data_path = '/home/ubuntu/Documents/v_e/data/frequency/{}_cleaned_{}_word_freqs{}.pkl'.format(country, period, s_flag)
    data_path = os.path.join(frequency_path,'{}_processed_json_{}_word_freqs{}.pkl'.format(country, period, s_flag))
    data = pd.read_pickle(data_path)
    freqs = [data.loc[word] for word in word_list if word in data.index]
    grp_freq = sum(freqs)
    return grp_freq

def rolling_z_score(freqs, window=8):
    def z_func(x):
        return (x[-1] - x[:-1].mean()) / x[:-1].std(ddof=0)
    return freqs.rolling(window=window+1).apply(z_func, raw=True)

def signif_change(freqs, window=8, direction=None):
    """
    find periods for which there was a significant change wrt the rolling average.

    freqs: (pd.Series) time series to check
    window: (int) number of periods prior to t over which to calc rolling mean/std/z_score
    direction: (str or NoneType) None for both signif increase and decrease, otherwise 'incr' or 'decr'
    """
    assert isinstance(freqs, pd.Series)
    z_scores = rolling_z_score(freqs, window)
    if not direction:
        result = z_scores[(z_scores >= 1.96) | (z_scores <= -1.96)]
    else:
        if 'incr' in direction:
            result = z_scores[z_scores >= 1.96]
        elif 'decr' in direction:
            result = z_scores[z_scores <= -1.96]
        else: 
            raise ValueError

    return result

def word_frequency(doc_list, **kwargs):
    """
    Get word frequency over a group of docs
    :param doc_list: list of full document file paths
    :param kwargs:
    :return: pandas Series of normed word frequency (per kword)
    """
    freqs = defaultdict(int)
    stream = SentStreamer(doc_list, language='en', **kwargs)
    for sent in stream:
        for tok in sent:
            freqs[tok] += 1

    # Normalize
    total_tokens = sum(freqs.values())
    normed_freqs = {k: v / total_tokens for k, v in freqs.items()}
    normed_freqs = pd.Series(normed_freqs)

    return normed_freqs


def list_crisis_docs(country, path,doc_data=None, period='crisis'):
    """
    Get list of docs that fall between beginning and peak crisis for country specified
    :param country: string. name of country to look for crisis docs
    :return: pandas df of docs
    """
    # Setup
    assert country in crisis_points.keys()
    assert period in ('crisis', 't-1', 't-2')
    data = doc_data if doc_data is not None else pd.read_pickle(os.path.join(path,"doc_details.pkl")) ## this is still not right, will fix latter

    # Tag docs as within crisis periods
    data['crisis'] = 0
    for i, (start, peak) in enumerate(zip(crisis_points[country]['starts'], crisis_points[country]['peaks'])):
        print("\rFinding crisis docs for period {} of {}".format(i + 1, len(crisis_points[country]['starts'])), end='')

        if period == 't-1':
            p = pd.Period(start)
            s = p.to_timestamp() - pd.DateOffset(years=1)
            s = s.to_period('M')
        elif period == 't-2':
            p = pd.Period(start)
            s = p.to_timestamp() - pd.DateOffset(years=2)
            s = s.to_period('M')
        else:
            s = pd.Period(start)
            p = pd.Period(peak)

        data.loc[(data['month'] >= s) & (data['month'] <= p), 'crisis'] = 1

    crisis_period_docs = id2full_path(data[data['crisis'] == 1].index,path)
    nocrisis_period_docs = id2full_path(data[data['crisis'] == 0].index,path)
    print("Filtering crisis docs")
    crisis = id2full_path([art['an'] for art in FileStreamer(crisis_period_docs, regions=[region[country]],
                                                              region_inclusive=True, title_filter=[country])],
                                                            path)
    return crisis


#def id2full_path(collection):
#    doc_path = "/home/ubuntu/Documents/v_e/cleaned"
#    
def id2full_path(collection,path):
    doc_path = path
    return [doc_path + os.path.sep + doc + ".json" for doc in collection]


def crisis_noncrisis_freqs(crisis, non_crisis, country, save=True, path="/home/ubuntu/Documents/v_e/data/frequency"):
    phraser_path = "/home/ubuntu/Documents/v_e/models/ngrams/2grams_default_10_20_NOSTOP"
    punct = '. , \' " ; : < > [ ] { } ( ) * & ^ % $ # @ ! = + - _ ` ~ | / / ?'.split(' ')
    stops = stopwords.words('english')  # + punct

    crisis_freqs = word_frequency(crisis, phraser=phraser_path, stopwords=stops,
                                  title_filter=[country], regions=[region[country]],
                                  region_inclusive=True, verbose=True)

    non_crisis_freqs = word_frequency(non_crisis, phraser=phraser_path, stopwords=stops,
                                      title_filter=[country], regions=[region[country]],
                                      region_inclusive=True, verbose=True)

    if save:
        crisis_freqs.to_pickle(path + os.path.sep + country + "_crisis_freqs.pkl")
        non_crisis_freqs.to_pickle(path + os.path.sep + country + "_non-crisis_freqs.pkl")

    return crisis_freqs, non_crisis_freqs


def freq_increasing(f1, f2):
    # get diff in relative frequency between f1 and f2
    diff = f2 - f1

    # Pull words which increased more than 2 stdevs more than mean diff
    increasing = diff[diff >= diff.mean() + 3 * diff.std()]

    return increasing


def word_clusters(words, model, k):
    mod = KeyedVectors.load(model)
    vecs = [mod.wv[w] for w in words]
    clusters = KMeans(n_clusters=k).fit(vecs)
    cluster_list = defaultdict(list)
    for w, c in zip(words, clusters.labels_):
        cluster_list[c].append(w)

    return cluster_list


if __name__ == '__main__':
    COUNTRY = 'argentina'

    crisis = pd.read_pickle("/home/ubuntu/Documents/v_e/data/frequency/{}_crisis_freqs.pkl".format(COUNTRY))
    non_crisis = pd.read_pickle("/home/ubuntu/Documents/v_e/data/frequency/{}_non-crisis_freqs.pkl".format(COUNTRY))
    country_freqs = pd.read_pickle("/home/ubuntu/Documents/v_e/data/frequency/{}_cleaned_month_word_freqs.pkl".format(COUNTRY))
    MODEL = "/home/ubuntu/Documents/v_e/models/vsms/word_vecs_5_10_200"

    increasing = freq_increasing(non_crisis, crisis)
    increasing.to_csv("/home/ubuntu/Documents/v_e/data/frequency/{}_crisis_words.csv".format(COUNTRY))
    increasing_clusters = word_clusters(increasing.index, model=MODEL, k=len(increasing) // 5)

    for i, (k, v) in enumerate(increasing_clusters.items()):
        fig = plot_frequency(country_freqs, words=v, country=COUNTRY)
        fig.suptitle("Monthly Word Frequency - {}".format(COUNTRY))
        fig.show()
        fig.savefig('/home/ubuntu/Documents/v_e/viz/{}_freqs_{}.png'.format(COUNTRY, i))
