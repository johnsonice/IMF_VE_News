import sys
sys.path.insert(0, './libs')
import gensim
from stream import SentStreamer_fast as SentStreamer
import random
import copy
import config


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == "__main__":

    # TODO Modularize w/argparse

    CORPUS = config.JSON_LEMMA
    OUT_DIR = config.VS_MODELS
    phraser = config.PHRASER
    
    size = 200
    window = 5
    min_count = 50
    workers = 4

    # Test
    use_skipgram = True
    
    random.seed(10)

    streamer = SentStreamer(CORPUS, language='en', phraser=phraser, stopwords=[], lemmatize=False, verbose=True)
    print(len(streamer.input_files))
    all_input_files = copy.deepcopy(streamer.input_files)
    random.shuffle(all_input_files)
    
    batched_files = list(chunks(all_input_files, config.SAMPLE_LIMIT))
    
    ## oneline training loop -- deal with large data size that can't fit into memory
    for batch_number, batch in enumerate(batched_files):
        print("Loading {}/{} batches, number of files: {}".format(i, len(batched_files), len(batch)))
        streamer.input_files = batch
        sentences = streamer.multi_process_files()
        if batch_number == 0:
            if use_skipgram:
                vectors = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count,
                                                 workers=workers, sg=1)  # Skipgram
            else:
                vectors = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count,
                                                 workers=workers)  # CBOW
        else:
            vectors.build_vocab(sentences, update=True)
            vectors.train(sentences, total_examples=vectors.corpus_count, epochs=vectors.epochs)
        
        del sentences
        print('sample results:')
        print(vectors.wv.most_similar('finance', topn=10))

    if use_skipgram:
        vectors.save(OUT_DIR + "/word_vecs_{}_{}_{}_{}".format(window, min_count, size, 'skipgram'))
    else:
        vectors.save(OUT_DIR + "/word_vecs_{}_{}_{}".format(window, min_count, size))

