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
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_lda(corpus, id2word_mapping, n_topics, clip, passes):
    corp = gensim.corpora.MmCorpus(corpus)
    if clip:
        clip = int(clip)
        corp = gensim.utils.ClippedCorpus(corp, clip) # LDA is slow-- use subsample for train
    corp_dict = gensim.corpora.Dictionary.load(id2word_mapping)
    lda_model = gensim.models.LdaMulticore(corp, num_topics=n_topics, 
                                           id2word=corp_dict, passes=passes, workers=4) 
    return lda_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corp_dir', default='../data/corpus_tfidf.mm')
    parser.add_argument('-d', '--dictionary', action='store', dest='dict_dir', default='../data/corpus_dict')
    parser.add_argument('-clip', '--clip', action='store', dest='clip', default=None)
    parser.add_argument('-top', '--n_topics', action='store', dest='n_top', type=int, default=100)
    parser.add_argument('-p', '--passes', action='store', dest='passes', type=int, default=4)
    parser.add_argument('-s', '--save', action='store', dest='save_dir', default='../data/topic/')
    opts = parser.parse_args()

    model = train_lda(opts.corp_dir, opts.dict_dir, opts.n_top, opts.clip, opts.passes)
    save_fname = 'lda_model_tfidf_{}_{}_{}'.format(opts.n_top, opts.clip, opts.passes)
    model.save(os.path.join(opts.save_dir, save_fname))

