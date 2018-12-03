import sys
sys.path.insert(0,'./libs')
import gensim
from stream import SentStreamer_fast as SentStreamer
import sys

CORPUS = '/data/News_data_raw/FT_WD/json_lemma'
OUT_DIR = "/data/News_data_raw/FT_WD/models/vsms"
phraser = '/data/News_data_raw/FT_WD/models/ngrams/2grams_default_10_20_NOSTOP'

size = 200
window = 5
min_count = 100
workers = 31

sentences = SentStreamer(CORPUS, language='en', phraser=phraser, lemmatize = False, verbose=True).multi_process_files()
vectors = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)
vectors.save(OUT_DIR + "/word_vecs_{}_{}_{}".format(window, min_count, size))


