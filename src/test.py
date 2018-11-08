#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 10:48:15 2018

@author: huang
"""

"""
Examine frequency of semantically related words in the corpus
"""
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
import pandas as pd
#from frequency_utils import plot_frequency
from plot_utils import plot_frequency
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from crisis_points import crisis_points
#%%
def crisis_plot(data, country=None, roll_avg=True, roll_window=20):
    """
    For any pandas time series, plot it and add annotations for crisis points given the country.
    :param data: pd.Series with PeriodIndex
    :param country: str
    :return: plt
    """
    assert isinstance(data, pd.Series)
    country = country.lower() if isinstance(country, str) else country
    plt.figure()
    if roll_avg:
        plot_data = data.rolling(window=roll_window).mean().values
    else:
        plot_data = data.values
    print('##1##')
    plt.plot(plot_data)
    plt_indices = [i for i in range(len(data)) if i % 2 == 0]
    plt_labs = list(data.index[plt_indices])
    print('##2##')
    plt.xticks(plt_indices, plt_labs, rotation=90, fontsize=6)
    plt.legend()
    crisis_starts = crisis_points[country]['starts']
    crisis_peaks = crisis_points[country]['peaks']
    print('##3##')
    for s, p in zip(crisis_starts, crisis_peaks):
        s_index = data.index.get_loc(s)
        p_index = data.index.get_loc(p)
        plt.axvline(x=s_index, color='grey', linestyle="--", linewidth=2)
        plt.axvline(x=p_index, color='red', linestyle="--", linewidth=2)
        plt.axvspan(s_index, p_index, facecolor='r', alpha=0.1)

    print('##4##')
    return plt.gcf()
#%%
def plot_similar_freqs(target_words, vecs, country, country_freqs, topn=10, roll_avg=True, roll_window=20):
    assert type(target_words) in (list, str)
    if vecs is not None:
        assert type(vecs) == Word2Vec

    # Find topn most similar words according to VSM
    target_words = [target_words] if isinstance(target_words, str) else target_words
#    words = [w[0] for w in vecs.most_similar(target_words, topn=topn)]
#    words += target

    if vecs is not None:
        try:
            words = [w[0] for w in vecs.most_similar(target_words, topn=topn)]
            words += target
        except:
            words = target_words
    else:
        words = target_words
    
    # Generate plot
    if not roll_avg:
        fig = plot_frequency(country_freqs, words=words, country=COUNTRY)
    else:
        word_freqs = [country_freqs.loc[word] for word in words if word in country_freqs.index and sum(country_freqs.loc[word] != 0)]
        grp_freq = sum(word_freqs)
        grp_rolling = grp_freq.rolling(window=roll_window).mean()
        fig = crisis_plot(grp_rolling, country=COUNTRY,roll_window=2)

    # return plot
    fig.suptitle("{} Frequency of top {} words most similar to {}".format(country, topn, target))
    return fig


if __name__ == '__main__':
    #MODEL = "/home/ubuntu/Documents/v_e/models/vsms/word_vecs_5_10_200"
    MODEL = None
    COUNTRY = 'argentina'
    if MODEL is not None:
        vecs = KeyedVectors.load(MODEL)
    else:
        vecs = None
        
    country_freqs = pd.read_pickle("../data/frequency/{}_processed_json_quarter_word_freqs.pkl".format(COUNTRY))

    target = 'warn fear worry'.split(" ")
    plot_similar_freqs(target, vecs=vecs, country=COUNTRY, country_freqs=country_freqs,
                       roll_avg=True, roll_window=2)
