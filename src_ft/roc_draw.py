import os
import pandas as pd
import sys
sys.path.insert(0, './libs')
import config
import matplotlib.pyplot as plt
from subprocess import Popen  # Allows to run other python files
import time

search_terms_file = 'grouped_search_words_final.csv'
sdf = pd.read_csv(os.path.join(config.SEARCH_TERMS, search_terms_file))

word_groups = list(sdf.index)

for word_group in word_groups:
    # set up
    base_fold = '/data/News_data_raw/FT_WD_research/threshold/'
    config.maybe_create(base_fold)
    class_type = 'Min1_AllCountry'
    countries = config.countries_just_five
    save_file = os.path.join(base_fold, 'cross_threshold_comparison_{}.csv'.format(word_group))
    word_group_df = pd.read_csv(save_file)

    for country in countries:
        sensitivties = word_group_df[country]['sensitivity']
        anti_sensitivties = [1-x for x in sensitivties]
        plt.scatter(anti_sensitivties, sensitivties)
        plt.title('ROC Curve {} on {}'.format(country, word_group))
        plt.xlabel('1-Sensitivity')
        plt.ylabel('Sensitivity')
        plt.savefig('/home/apsurek/IMF_VE_News/research/ROC_curves/{}_on_{}.png'.format(country, word_group))
        plt.show()
        plt.clf()

