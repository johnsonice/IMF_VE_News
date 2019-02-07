#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:59:43 2019
@author: chuang
"""
"""
convert text corpus into tfidf bow format
-- creates dict mapping between text words and int keys
-- saves corpus in gensim bow format
-- saves tfidf transformation of bow corpus
"""
import sys
sys.path.insert(0,'./libs')
import os 
import gensim
from stream import DocStreamer_fast
import argparse
import config
import pickle


def create_dict(stream, min_count=20, max_ratio=0.4, keep_n=500000, verbose=False):
    if verbose:
        print("Creating Dictionary...")
    corpus_dict = gensim.corpora.Dictionary(stream)
    corpus_dict.filter_extremes(no_below=min_count, no_above=max_ratio, keep_n=keep_n)
    return corpus_dict


def corp_to_bow(stream, corp_dict, verbose=False):
    if verbose:
        print("\nConverting to BOW")
    corpus_bow = [corp_dict.doc2bow(doc) for doc  in stream]
    return corpus_bow


def bow_to_tfidf(corp_bow, verbose=False):
    if verbose:
        print("\nConverting to tfidf")
    tfidf = gensim.models.TfidfModel(corp_bow)
    corpus_tfidf = tfidf[corp_bow]
    return corpus_tfidf

class args_class(object):
    def __init__(self, in_dir=config.JSON_LEMMA_SMALL,out_dir=config.BOW_TFIDF_DOCS,phraser=config.PHRASER,
                 lemmatize=False,lang='en',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.phraser = phraser
        self.lang = lang
        self.lemmatize = lemmatize
        self.verbose = verbose
        
        
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
        parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
        parser.add_argument('-p', '--phraser', action='store', dest='phraser', default=None)
        parser.add_argument('-q', '--quiet', action='store_false', dest='verbose', default=True)
        parser.add_argument('-l', '--lemmatize', action='store_true', dest='lemmatize', default=True)
        parser.add_argument('-lang', '--language', action='store', dest='lang', default='en')
        args = parser.parse_args()
    except:
        args = args_class()
    
    # chec if pickle is already produced 
    # Setup the streamer
    ## stop words should already be deleted 
    pickle_data_file = os.path.join(args.out_dir,'all_docs.p')
    if not os.path.isfile(pickle_data_file):
        if args.verbose:
            print('Reading documents form json ...')
        stream = DocStreamer_fast(args.in_dir, language=args.lang, phraser=args.phraser, 
                                  verbose=args.verbose, lemmatize=args.lemmatize,stopwords=None).multi_process_files(chunk_size=5000)
    else:
        if args.verbose:
            print('Reading documents form pickle ...')
        stream = pickle.load(open(pickle_data_file,'rb'))
    
    
    # Create dictionary, bow corpus, and tfidf corpus
    dictionary = create_dict(stream, verbose=args.verbose)
    bow = corp_to_bow(stream, dictionary, verbose=args.verbose)
    tfidf = bow_to_tfidf(bow, verbose=args.verbose)

    # Save dictionary, bow corpus, and tfidf corpus
    dictionary.save(os.path.join(args.out_dir, 'dictionary'))
    gensim.corpora.MmCorpus.serialize(os.path.join(args.out_dir, 'bow.mm'), bow)
    gensim.corpora.MmCorpus.serialize(os.path.join(args.out_dir, 'tfidf.mm'), tfidf)
