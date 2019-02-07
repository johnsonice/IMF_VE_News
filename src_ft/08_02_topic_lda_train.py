"""
topic_lda_train.py

description: Used to train lda topic models

usage: python3 topic_lda_train.py
note: see optional args with --help flag
"""

import os
import gensim
import argparse
import logging
import config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_lda(corpus, id2word_mapping, n_topics, clip, passes):
    corp = gensim.corpora.MmCorpus(corpus)
    if clip:
        clip = int(clip)
        corp = gensim.utils.ClippedCorpus(corp, clip) # LDA is slow-- use subsample for train
    corp_dict = gensim.corpora.Dictionary.load(id2word_mapping)
    lda_model = gensim.models.LdaMulticore(corp, num_topics=n_topics, eta='auto',
                                           id2word=corp_dict, passes=passes, workers=int(os.cpu_count()/2)-1) 
    
    ## alpha = 'auto' ## better results, but not available for multi core
    
    return lda_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corp_dir', default=os.path.join(config.BOW_TFIDF_DOCS,'tfidf.mm'))
    parser.add_argument('-d', '--dictionary', action='store', dest='dict_dir', default=os.path.join(config.BOW_TFIDF_DOCS,'dictionary'))
    parser.add_argument('-clip', '--clip', action='store', dest='clip', default=None)
    parser.add_argument('-top', '--n_topics', action='store', dest='n_top', type=int, default=100)
    parser.add_argument('-p', '--passes', action='store', dest='passes', type=int, default=4)
    parser.add_argument('-s', '--save', action='store', dest='save_dir', default=config.TOPIC_MODELS)
    args = parser.parse_args()


    model = train_lda(args.corp_dir, args.dict_dir, args.n_top, args.clip, args.passes)
    save_fname = 'lda_model_tfidf_{}_{}_{}'.format(args.n_top, args.clip, args.passes)
    model.save(os.path.join(args.save_dir, save_fname))

    ## test if model can be loade properly
    #del model 
    
