import gensim
from stream import SentStreamer

CORPUS = '/home/ubuntu/Documents/v_e/cleaned'
OUT_DIR = "/home/ubuntu/Documents/v_e/models/vsms"

size = 200
window = 5
min_count = 10
workers = 8
iter = 3

phraser = '/home/ubuntu/Documents/v_e/models/ngrams/2grams_default_10_20_NOSTOP'
sentences = SentStreamer(CORPUS, language='en', phraser=phraser, verbose=True)

vecs = gensim.models.fasttext.FastText(sentences=sentences, window=window, min_count=min_count, workers=workers, iter=iter)
vecs.save(OUT_DIR + "/fast_text_vecs_{}_{}_{}".format(window, min_count, size))
