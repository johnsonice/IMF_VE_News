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
import ujson as json

model_folder = "/data/News_data_raw/FT_WD/models/topics"
this_model = "lda_model_tfidf_100_None_4"
model_address = os.path.join(model_folder,this_model)
loaded_model = gensim.models.ldamodel.LdaModel.load(model_address)
print("MODEL LOADED at:", model_address)

test_doc_folder = "/data/News_data_raw/Financial_Times_processed/FT_json_current"
this_doc = "ft-1cd924b9-7579-3662-869e-331646fbae2e.json"
doc_address = os.path.join(test_doc_folder, this_doc)
with open(doc_address, 'r', encoding="utf-8") as f:
    json_loaded = json.loads(f.read())

print("Loaded doc at:",doc_address)
print("Loaded json file type:",type(json_loaded))
print("Loaded json keys:",json_loaded.keys())

doc_text = json_loaded['body']
#print("BODY:",doc_text)

tokens = doc_text.split()
print("Doc tokenized")

corpus_path = os.path.join(config.BOW_TFIDF_DOCS,'tfidf.mm')
corpus = gensim.corpora.MmCorpus(corpus_path)
print("Corpus loaded")

common_dictionary_path = os.path.join(config.BOW_TFIDF_DOCS,'dictionary')
common_dictionary = gensim.corpora.Dictionary.load(common_dictionary_path)
print("Dictionary loaded")

bowed = common_dictionary.doc2bow(tokens)
print("Doc BOWED")

topic_props = loaded_model.get_document_topics(bowed,minimum_probability=0)
print("Doc predicted on")

topic_props_df = pd.DataFrame(topic_props)
print(topic_props_df)
