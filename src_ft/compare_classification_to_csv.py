import sys
sys.path.insert(0,'./libs')
import argparse
from gensim.models.keyedvectors import KeyedVectors
#from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
import pandas as pd
#import numpy as np
import os
from mp_utils import Mp
import config

class_type_setups = config.class_type_setups

base_eval_path = config.EVAL_WG
combined_df = pd.DataFrame()
eval_types = ['grouped_words'] #TEMP
for e_type in eval_types:
    for i in range(len(class_type_setups)):
        class_type = class_type_setups[i][0]
        folder_path = os.path.join(base_eval_path, class_type)
        file_path = os.path.join(folder_path, e_type, 'overall_agg_sim_True_overall_month_offset_{}_smoothwindow_'
                                                      '{}_evaluation.csv'.format(config.months_prior,
                                                                                 config.smooth_window_size))
        base_df = pd.read_csv(file_path)
        print("read df from", file_path)

        word_list = list(base_df['word'])
        new_df_dict = {'classification_type': [class_type]}

        for j in range(len(word_list)):
            word = word_list[i]
            new_df_dict[word+'_recall'] = base_df['recall'][j]
            new_df_dict[word+'_prec'] = base_df['prec'][j]
            new_df_dict[word+'_f2'] = base_df['f2'][j]

        app_df = pd.DataFrame(new_df_dict)
        combined_df = combined_df.append(app_df)
        print("Appended df", app_df)

    out_file = os.path.join(base_eval_path,'classification_comparison',
                            'country_classification_comparison_using_{}.csv'.format('grouped_words'))# TEMP

    try:
        already_written = pd.read_csv(out_file)
        combined_df = already_written.append(combined_df)
        print("Adding to previous csv")
    except IOError:
        pass

    combined_df.to_csv(out_file)
    print("Saved dataframe in file {}".format(out_file))

