import os
import pandas as pd
import sys
sys.path.insert(0, './libs')
import config
from subprocess import Popen  # Allows to run other python files

sample_size = 50000  # Test
# sample_size = 500000
window_sizes = [5]
#window_sizes = [5, 10, 15, 20]
models_directory = '/data/News_data_raw/FT_WD/models/vsms_test'

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
