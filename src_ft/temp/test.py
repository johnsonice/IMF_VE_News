import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'../libs')
from stream import FileStreamer_fast as FileStreamer
import ujson as json
from nltk.tokenize import word_tokenize
import spacy 
from spacy.symbols import ORTH, LEMMA, POS, TAG
nlp = spacy.load("en_core_web_lg",disable=['tagger','ner','parser','textcat'])
special_case = [{ORTH:u'__NUMBER__',LEMMA:u'__NUMBER__'}]
nlp.tokenizer.add_special_case(u'__NUMBER__',special_case)


#%%
class args_class(object):
    def __init__(self, in_dir,out_dir,period='crisis',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.period=period
        self.verbose = verbose

args = args_class('/data/News_data_raw/FT_WD/json_lemma','../../data/doc_meta', verbose = False)

#%%
streamer = FileStreamer(args.in_dir,verbose=True)
files = streamer.input_files
#%%

with open(files[10], 'r', encoding="utf-8") as f:
    data = json.loads(f.read())

#%%
## used mata fields 
## 'language_code' 'region_codes''snippet''title' 'body' 'an' 'publication_date'
    
meta_list = ['language_code','region_codes','snippet','title','an','publication_date'] #,'body'

for m in meta_list:
    print('{}: {}'.format(m,data[m]))
    
