import os
import pandas as pd
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config


import gensim

corpus_path = os.path.join(config.BOW_TFIDF_DOCS, 'tfidf.mm')
corpus = gensim.corpora.MmCorpus(corpus_path)

common_dictionary_path = os.path.join(config.BOW_TFIDF_DOCS, 'dictionary')
common_dictionary = gensim.corpora.Dictionary.load(common_dictionary_path)

model_folder = "/data/News_data_raw/FT_WD/models/topics"
this_model = "lda_model_tfidf_100_None_4"
model_address = os.path.join(model_folder, this_model)
loaded_model = gensim.models.ldamodel.LdaModel.load(model_address)

topics = range(100)
pd_dict = {}
for top in topics:
    these_topic_word = loaded_model.show_topic(top, topn=30)
    as_list = [x[1] for x in these_topic_word]
    pd_dict.update({top: as_list})

df = pd.DataFrame(pd_dict)
df.to_csv('/home/apsurek/IMF_VE_News/csv_out/lda_100_topic_words')
print('Saved df')