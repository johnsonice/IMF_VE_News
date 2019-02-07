"""
Light preprocessing for corpus
"""
import sys
sys.path.insert(0,'./libs')
import config
import os
#from glob import glob
#import re
import ujson as json
from mp_utils import Mp
from stream import SentStreamer_fast as SentStreamer
from spacy.lang.en.stop_words import STOP_WORDS as stops
#from string import punctuation as punct
bad_char = ['!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','`','{','|','}','~','NUMBER','=','-','«','.']


#%%

def deep_clean(token):
    
    if len(token)<=3:
        return False
    
    if len(token)>20:
        return False
    
    if token in stops:
        return False

    for b in bad_char:
        if b in token:
            return False
    
    return True

def preprocess(json_article,lemma = True):
    try:
        text = json_article['body']
        tokens = text.split(' ')
        tokens = [t for t in tokens if deep_clean(t)]
        
        text = ' '.join(tokens)
        json_article['body'] = text
        return json_article
    except Exception:
        print('no text in article {}'.format(json_article['an']))
        return False        


class args_class(object):
    def __init__(self, in_dir,out_dir,verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.verbose = verbose
#%%
if __name__ == '__main__':

    args = args_class(config.JSON_LEMMA,config.JSON_LEMMA_SMALL, verbose = True)
        
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
    res = mp.multi_process_files(workers=20,chunk_size=10000)
    


