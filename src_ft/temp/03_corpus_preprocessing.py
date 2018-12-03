"""
Light preprocessing for corpus
"""
import sys
sys.path.insert(0,'./libs')
import os
from glob import glob
import re
import ujson as json
import argparse
from mp_utils import Mp
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stops
import itertools
import copy
import time

nlp = spacy.load("en_core_web_lg",disable=['tagger','ner','parser','textcat'])

#nlp = spacy.load("en")
#nlp.remove_pipe('tagger')
#nlp.remove_pipe("ner")

#%%
def preprocess(json_article):
    try:
        text = json_article['body']
        # Normalize capitalization
        #text = text.lower()
        
        # Normalize spacing
        text = re.sub("\s+", " ", text)
        
        # Normalize numbers (the fact that a number appears may be important, but not the actual number)
        text = re.sub("(\d+[,./]?)+", "<<NUMBER>>", text)
        
        json_article['body'] = text
        return json_article
    except Exception:
        print('no text in article {}'.format(json_article['an']))
        return False

def read_jsons(flist):
    for fname in flist:
        try:
            with open(fname, 'r') as f:
                fj = preprocess(json.loads(f.read()))
            if fj and len(fj['body'])>0:
                fj_meta = copy.copy(fj)
                fj_meta['body']= None
                yield fj['body'],fj_meta
            else:
                continue
        except:
            print(fname)
            continue
        
def punct_space(token):
    return token.is_punct or token.is_space

class args_class(object):
    def __init__(self, in_dir,out_dir,verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.verbose = verbose

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
        parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
        parser.add_argument('-o', '--verbose', action='store', dest='verbose', default=False)
        args = parser.parse_args()
    except:
        ## give some default arguments
        args = args_class('/data/News_data_raw/Financial_Times_processed/FT_json_historical','/data/News_data_raw/FT_WD/json_lemma', verbose = True)
        
    #flist = glob(args.in_dir + '/*.json')
    #fl_len = len(flist)
    
    #%%
    ## grab all files 
    flist = glob(args.in_dir + '/*.json')[:5000]
    fl_len = len(flist)
    print('Total number of files to process {}.\n'.format(fl_len))
    
    ## create generators
    g_text,g_meta = itertools.tee(read_jsons(flist))
    del flist
    texts = (t for (t,m) in g_text )
    metas = (m for (t,m) in g_meta )
    docs = nlp.pipe(texts,batch_size=5000,n_threads=30)
    
    ## process and dump processed files
    start = time.time()
    counter = 0 
    for m,t in zip(metas,docs):
        counter+=1
        if counter%100==0:
            end = time.time()  
            time_used = time.strftime('%H:%M:%S', time.gmtime(end - start))
            print('processed {}/{} files. Time used up to now: {}'.format(counter,fl_len,time_used))
        try:
            #toks = [tok.lemma_ for tok in t if not punct_space(tok)]
            toks = [tok.lemma_ for tok in t]
            if len(toks)>0:
                st = ' '.join(toks)
                m['body'] = st
                outf = os.path.join(args.out_dir, m['an']+'.json')
                with open(outf, 'w') as f:
                    f.write(json.dumps(m))
        except:
            print(m['an'])
    
    end = time.time()  
    time_used = time.strftime('%H:%M:%S', time.gmtime(end - start))
    print('Finished. total time used: {}'.format(time_used))
    

