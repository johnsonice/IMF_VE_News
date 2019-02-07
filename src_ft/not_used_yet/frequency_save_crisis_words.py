"""
For a given country, save csv of those words which increase significantly during crisis periods.
"""
from frequency_utils import freq_increasing, word_clusters
from plot_utils import plot_frequency
from crisis_points import crisis_points
import pandas as pd
import sys


if __name__ == '__main__':
    assert len(sys.argv) <= 2
    if len(sys.argv) == 2:
        countries = sys.argv[1] # calculate for list of countries supplied on command line, else for all
        assert isinstance(countries, list)
    else:
        countries = crisis_points.keys()

    for country in countries:
        crisis = pd.read_pickle("/home/ubuntu/Documents/v_e/data/frequency/{}_crisis_freqs.pkl".format(country))
        non_crisis = pd.read_pickle("/home/ubuntu/Documents/v_e/data/frequency/{}_non-crisis_freqs.pkl".format(country))
        country_freqs = pd.read_pickle("/home/ubuntu/Documents/v_e/data/frequency/{}_cleaned_month_word_freqs.pkl".format(country))
        MODEL = "/home/ubuntu/Documents/v_e/models/vsms/word_vecs_5_10_200"

        increasing = freq_increasing(non_crisis, crisis)
        increasing.to_csv("/home/ubuntu/Documents/v_e/data/frequency/{}_crisis_words.csv".format(country))
