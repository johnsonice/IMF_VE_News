import os
import pandas as pd
import sys
sys.path.insert(0, './libs')
import config
from subprocess import Popen  # Allows to run other python files
import time

# set up
base_fold = '/data/News_data_raw/FT_WD_research/threshold/'
config.maybe_create(base_fold)
class_type = 'Min1_AllCountry'
# countries = config.countries
countries = ['argentina']  # TEMP
word_groups = ['all_language']  # TEMP
search_terms = 'grouped_search_words_final.csv'
top_n = 15

thresh_values = [0, 1.645, 1.96, 2.576, 99]
thresh_files = []
for thresh_value in thresh_values:
    new_folder = os.path.join(base_fold, str(thresh_value))
    config.maybe_create(new_folder)
    thresh_files.append(new_folder)
    Popen(['python', '07_02_frequency_eval_aggregate.py', '-z', '{}'.format(thresh_value), '-ep',
           '{}'.format(new_folder), '-c', '{}'.format("".join(countries)), '-gsf', '{}'.format(search_terms), '-tn',
           '{}'.format(top_n)])

summ_dict = {}
summ_df = pd.DataFrame()
fscores = []
tps = []
fps = []
fns = []
sensitivities = []

for word_group in word_groups:
    df_index = pd.MultiIndex.from_product([thresh_values, countries], names=['threshold', 'country'])
    word_group_df = pd.DataFrame(index=df_index, columns=['fscore', 'tp', 'fp', 'fn', 'sensitivity'])
    waiting = True
    while waiting:
        try:
            for thresh_value in thresh_values:

                        read_file = 'agg_sim_True_month_offset_24_smoothwindow_18_{}_evaluation.csv'.format(word_group)
                        read_path = os.path.join(base_fold, str(thresh_value), read_file)
                        read_df = pd.read_csv(read_path,
                                              index_col='Unnamed: 0')
                        waiting = False
                        for country in countries:
                            # Save threshold country prediction info, calculate sensitivity of prediction
                            country_slice = read_df.loc[country]
                            tp = country_slice['tp']
                            fn = country_slice['fn']
                            sensitivity = tp/(tp+fn)
                            word_group_df.loc[(thresh_value, country)] = pd.Series({'fscore': country_slice['fscore'], 'tp': tp,
                                                                                    'fp': country_slice['fp'], 'fn': fn,
                                                                                    'sensitivity': sensitivity})
        except IOError:
            time.sleep(30)
            print("Waiting up on 07_02")
            waiting = True

    save_file = os.path.join(base_fold, 'cross_threshold_comparison_{}.csv'.format(word_group))
    word_group_df.to_csv(save_file)



