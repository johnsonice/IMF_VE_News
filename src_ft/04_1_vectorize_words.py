import sys
sys.path.insert(0,'./libs')
import gensim
from stream import SentStreamer_fast as SentStreamer
import sys
import random
import copy
#from random import shuffle
import config

#%%
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":
    
    CORPUS = config.JSON_LEMMA
    OUT_DIR = config.VS_MODELS
    phraser = config.PHRASER
    
    size = 200
    window = 5
    min_count = 50
    workers = 4
    
    random.seed(10)

    
    streamer = SentStreamer(CORPUS, language='en', phraser=phraser,stopwords=[], lemmatize = False, verbose=True)
    print(len(streamer.input_files))
    all_input_files = copy.deepcopy(streamer.input_files)
    random.shuffle(all_input_files)
    
    batched_files = list(chunks(all_input_files,config.SAMPLE_LIMIT))
    
    ## oneline training loop -- deal with large data size that can't fit into memory
    for i,batch in enumerate(batched_files):
        print("Loading {}/{} batches, number of files: {}".format(i,len(batched_files),len(batch)))
        streamer.input_files = batch
        sentences = streamer.multi_process_files()
        if i == 0: 
            vectors = gensim.models.Word2Vec(sentences, size=size, window=window, min_count=min_count, workers=workers)
        else:
            vectors.build_vocab(sentences,update=True)
            vectors.train(sentences,total_examples=vectors.corpus_count,epochs=vectors.epochs)
        
        del sentences
        print('sample results:')
        print(vectors.wv.most_similar('finance', topn=10))
        
    vectors.save(OUT_DIR + "/word_vecs_{}_{}_{}".format(window, min_count, size))
