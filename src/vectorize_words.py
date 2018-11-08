import gensim
import pandas
from stream import SentStreamer_fast as SentStreamer
import sys

CORPUS = '../cleaned_small'
OUT_DIR = "../models/vsms"
phraser = '../models/ngrams/2grams_default_10_20_NOSTOP'

size = 200
window = 5
min_count = 10
workers = 4

sentences = SentStreamer(CORPUS, language='en', phraser=phraser, verbose=True)
vectors = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)
vectors.save(OUT_DIR + "/word_vecs_{}_{}_{}_lemmatized".format(window, min_count, size))


