# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import os 
import config
#%%
common_dictionary = Dictionary.load(os.path.join(config.BOW_TFIDF_DOCS,'dictionary'))
common_cropus = [common_dictionary.doc2bow(text) for text in common_texts]
common_cropus = common_cropus *100
lda = LdaModel(common_cropus, num_topics = 10)
lda.show_topics()
#%%

import pyLDAvis.gensim
import pyLDAvis
#%%

viz_data = pyLDAvis.gensim.prepare(lda,common_cropus,common_dictionary)
pyLDAvis.save_html(viz_data,"ldaviz_t{}.html".format(10))