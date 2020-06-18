import sys
sys.path.insert(0,'../src_ft/libs')
sys.path.insert(0,'../src_ft')
import argparse
from gensim.models.keyedvectors import KeyedVectors
#from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
import pandas as pd
#import numpy as np
import os
from mp_utils import Mp
import config
import gensim
import argparse
import config
import os
import pandas as pd
from topic_model_utils import topic2df ##print_topics_gensim

model_folder = "/data/News_data_raw/FT_WD/models/topics"
this_model = 'lda_model_tfidf_100_None_4'
model_address = os.path.join(model_folder,this_model)
loaded_model = gensim.models.ldamodel.LdaModel.load(model_address)
print("MODEL LOADED at:", model_address)
