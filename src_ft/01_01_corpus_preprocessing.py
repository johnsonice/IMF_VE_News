"""
Light preprocessing for corpus
"""
import sys
sys.path.insert(0,'./libs')
import config
import os
from glob import glob
import re
import ujson as json
from mp_utils import Mp
from stream import SentStreamer_fast as SentStreamer
import spacy
#from spacy.lang.en.stop_words import STOP_WORDS as stops
nlp = spacy.load("en_core_web_lg",disable=['tagger','ner','parser','textcat'])
#nlp.remove_pipe('tagger')
#nlp.remove_pipe("ner")
from spacy.symbols import ORTH, LEMMA, POS, TAG
special_case = [{ORTH:u'__NUMBER__',LEMMA:u'__NUMBER__'}]
nlp.tokenizer.add_special_case(u'__NUMBER__',special_case)

#%%
def preprocess(json_article,lemma = True):
    try:
        text = json_article['body']
        # Normalize capitalization
        #text = text.lower()
        
        # Normalize spacing
        text = re.sub("\s+", " ", text)
        text = re.sub("\'+", " ", text)
        #text = re.sub("\.+", " ", text)
        
        # Normalize numbers (the fact that a number appears may be important, but not the actual number)
        text = re.sub("([',./]?\d+[',./]?)+", " __NUMBER__ ", text)
        
        # lemmentize 
        if lemma:
            toks = nlp(text)
            toks = [tok.lemma_ for tok in toks if not tok.is_space]
            text = ' '.join(toks)
            
        json_article['body'] = text
        return json_article
    except Exception:
        print('no text in article {}'.format(json_article['an']))
        return False        
        
def punct_space(token):
    return token.is_punct or token.is_space

class args_class(object):
    def __init__(self, in_dir,out_dir,verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.verbose = verbose

if __name__ == '__main__':

    args = args_class(config.RAW_DATA_PATH,config.JSON_LEMMA, verbose = True)
        
    ## grab all files 
    flist = SentStreamer(args.in_dir).input_files
    fl_len = len(flist)
    print('Total number of files to process {}.\n'.format(fl_len))

    ## process and dump processed files
    def process_jsons(fname,out_dir=args.out_dir):
        try: 
            with open(fname, 'r') as f:
                fj = preprocess(json.loads(f.read()))
            if fj and len(fj['body'])>0:
                outf = os.path.join(args.out_dir, fj['an']+'.json')
                with open(outf, 'w') as _f:
                    _f.write(json.dumps(fj))
                #print('file produced')
        except:
            print(fname)
            
    ## multi process files 
    mp = Mp(flist,process_jsons) 
    res = mp.multi_process_files(workers=25,chunk_size=5000)
    


