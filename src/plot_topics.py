"""
Examine frequency of semantically related words in the corpus
"""
import gensim
import pandas as pd
from plot_utils import plot_frequency
from crisis_points import crisis_points
import argparse


def plot_topic(topic_n, model, country, country_freqs, topn=10, aggregate=True, roll_avg=True,
                       roll_window=8, z_score=True, slope=True, anomalies=True):

    terms = [t[0] for t in model.show_topic(topic_n, topn=15)]

    # Generate plot
    fig = plot_frequency(country_freqs, words=terms, country=country, roll_avg=roll_avg,
                         roll_window=roll_window, aggregate=aggregate, z_score=z_score, slope=slope, anomalies=True)

    # return plot
    fig.suptitle("{} Frequency of top {} words most similar to {}".format(country, topn, terms))
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--lda_model', action='store', dest='model',
                        default="../data/topic/lda_model_tfidf_500_None_4")
    parser.add_argument('-top', '--topics', nargs='+', action='store', dest='topics')
    parser.add_argument('-c', '--countries', action='store', dest='countries', default=','.join(crisis_points.keys()),
                        help='comma separated list of countries')
    parser.add_argument('-p', '--period', action='store', dest='period', default='quarter')
    opts = parser.parse_args()

    countries = opts.countries.split(',')
    topics = [int(t) for t in opts.topics]
    model = gensim.models.ldamodel.LdaModel.load(opts.model)
    roll_window = 8
    z_score = False
    slopes = False

    for country in countries:
        print("\rworking on {}...".format(country), end=" ")
        try:
            country_data = pd.read_pickle(
                "../data/frequency/{}_cleaned_{}_word_freqs.pkl".format(country, opts.period))
            for topic in topics:
                fig = plot_topic(topic, topn=15, model=model, country=country,
                                         country_freqs=country_data.loc[:, :'1999Q2'],
                                         roll_avg=True, roll_window=roll_window, aggregate=True, z_score=z_score,
                                         slope=slopes)
                fig.set_size_inches(30, 15)
                zflag = '_zscore' if z_score else ''
                slopeflag = '_slopes' if slopes else ''
                fig.savefig('../viz/{}_cumfreqs_topic{}_window={}'.format(country, topic, zflag, slopeflag, roll_window))
                fig.clf()
        except Exception as e:
            print('{}: {}'.format(country, e))
            continue
