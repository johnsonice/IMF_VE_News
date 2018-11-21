from plot_utils import crisis_plot
from matplotlib import pyplot as plt
import pandas as pd
import os
from numpy import nan


def sentiment_plot(country_data, country, normed=False):
    """
    :param country_data: Pandas Series with monthly period as the index and sentiment as the values
    :param country: string or NoneType. This is the key used to identify crisis periods for the plot.
    :param normed: Boolean. If true, country sentiment is normed against sentiment over entire corpus for each time period.
    :return: pyplot figure
    """

    if normed:
        norm_data = pd.read_pickle('/home/ubuntu/Documents/v_e/data/sentiment/monthly_corpus_sentiment.pkl')
        norm_idx = pd.PeriodIndex(start=norm_data.index[0], end=norm_data.index[-1], freq='M')
        norm_data = norm_data.reindex(norm_idx, fill_value=nan)

        data = country_data.reindex(norm_idx, fill_value=nan)
        country_data = data - norm_data
    else:
        idx = pd.PeriodIndex(start=country_data.index[0], end=country_data.index[-1], freq='M')
        country_data = country_data.reindex(idx, fill_value=nan)

    fig = crisis_plot(country_data, country)
    return fig

if __name__ == '__main__':
    dataf = '/home/ubuntu/Documents/v_e/data/sentiment/mexico_sentiment.pkl'
    sent = pd.read_pickle(dataf)
    #sent = sent.groupby('month')['sentiment'].mean()

    #country = os.path.basename(dataf).replace("_sentiment.pkl", "")
    country = os.path.basename(dataf).replace("_sentiment.pkl", "")
    f = sentiment_plot(sent, country, normed=True)
    f.suptitle("Monthly Sentiment across entire corpus")
    plt.axhline(0, color='k')
