import os
import pandas as pd
import sys
sys.path.insert(0, './libs')
import config
from subprocess import Popen  # Allows to run other python files

#sample_size = 50000  # Test
sample_size = 500000
#window_sizes = [5]
window_sizes = [5, 10, 15, 20]
min_count = 50
size = 200
top_n = 15
search_terms_file = 'grouped_search_words_final.csv'
models_directory = '/data/News_data_raw/FT_WD_research/models/vsms_test'
config.maybe_create(models_directory)
''' TEMP
for window_size in window_sizes:
    call_list = ['python', '04_1_vectorize_words.py', '-wd', str(window_size), '-spz', str(sample_size),
                 '-out', models_directory]
    # Run CBOW
    child = Popen(call_list)
    child.wait()

    # Run Skipgram
    call_list.append('-skg')
    child = Popen(call_list)
    child.wait()
'''
results_folder = '/data/News_data_raw/FT_WD_research/models/vsms_test/results'
config.maybe_create(results_folder)
for window_size in window_sizes:
    calls = []
    cbow_model = "/data/News_data_raw/FT_WD_research/models/vsms_test/word_vecs_{}_{}_{}".format(window_size, min_count, size)
    #cbow_model = "/data/News_data_raw/FT_WD/models/vsms_test/word_vecs_{}_{}_{}".format(window_size, min_count, size)
    cbow_fold = os.path.join(results_folder, str(window_size)+'_cbow')
    config.maybe_create(cbow_fold)
    cbow_call = ['python', '07_02_frequency_eval_aggregate.py', '-ep', '{}'.format(cbow_fold),
                 '-gsf', '{}'.format(search_terms_file), '-tn', '{}'.format(top_n), '-wv', cbow_model]
    calls.append(cbow_call)

    skg_model = "/data/News_data_raw/FT_WD_research/models/vsms_test/word_vecs_{}_{}_{}_{}".format(window_size, min_count, size, 'skipgram')
    #skg_model = "/data/News_data_raw/FT_WD/models/vsms_test/word_vecs_{}_{}_{}_{}".format(window_size, min_count, size, 'skipgram')
    skg_fold = os.path.join(results_folder, str(window_size)+'_skipgram')
    config.maybe_create(skg_fold)
    skg_call = ['python', '07_02_frequency_eval_aggregate.py', '-ep', '{}'.format(skg_fold),
                '-gsf', '{}'.format(search_terms_file), '-tn', '{}'.format(top_n), '-wv', skg_model]
    calls.append(skg_call)
    for call in calls:
        child = Popen(call)
        child.wait()
