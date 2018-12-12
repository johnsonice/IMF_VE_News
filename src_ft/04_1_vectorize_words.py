import sys
sys.path.insert(0,'./libs')
import gensim
from stream import SentStreamer_fast as SentStreamer
import sys
import config

CORPUS = '/data/News_data_raw/FT_WD/json_lemma'
OUT_DIR = "/data/News_data_raw/FT_WD/models/vsms"
phraser = '/data/News_data_raw/FT_WD/models/ngrams/2grams_default_10_20_NOSTOP'

CORPUS = config.JSON_LEMMA
OUT_DIR = config.VS_MODELS
phraser = config.PHRASER

size = 200
window = 5
min_count = 50
workers = 31

streamer = SentStreamer(CORPUS, language='en', phraser=phraser,stopwords=[], lemmatize = False, verbose=True)
if len(streamer.input_files) > config.SAMPLE_LIMIT:
    streamer.input_files = streamer.input_files[-config.SAMPLE_LIMIT:]
    print("number of input files is too large: only load last {}".format(config.SAMPLE_LIMIT))
    
sentences = streamer.multi_process_files()

vectors = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)
vectors.save(OUT_DIR + "/word_vecs_{}_{}_{}".format(window, min_count, size))


