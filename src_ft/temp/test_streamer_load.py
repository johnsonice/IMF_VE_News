import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config
from stream import FileStreamer_fast as FileStreamer
from stream import DocStreamer_fast as DocStreamer
import ujson as json
from nltk.tokenize import word_tokenize
import pickle 

#import spacy 
#from spacy.symbols import ORTH, LEMMA, POS, TAG
#nlp = spacy.load("en_core_web_lg",disable=['tagger','ner','parser','textcat'])
#special_case = [{ORTH:u'__NUMBER__',LEMMA:u'__NUMBER__'}]
#nlp.tokenizer.add_special_case(u'__NUMBER__',special_case)


#%%
class args_class(object):
    def __init__(self, in_dir,out_dir,period='crisis',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.period=period
        self.verbose = verbose

args = args_class(config.JSON_LEMMA,config.AUG_DOC_META_FILE, verbose = False)
#%%
args.in_dir = os.path.join(config.PROCESSING_FOLDER,'json_lemma_small')
streamer = DocStreamer(args.in_dir, 
                       language='en',
                       phraser=config.PHRASER,
                       stopwords=None, 
                       lemmatize=False)#.multi_process_files(workers=2,chunk_size = 100))
streamer.input_files = streamer.input_files

docs= streamer.multi_process_files(workers =30,chunk_size =5000)


#%%
with open('test.p','wb') as f:
    pickle.dump(docs,f)
    
#%%
#with open('test.p','rb') as f:
#    test2 = pickle.load(f)


#%%
#streamer = FileStreamer(args.in_dir,verbose=True)
#files = streamer.input_files
##%%
#
#with open(files[10002], 'r', encoding="utf-8") as f:
#    data = json.loads(f.read())
#data['body']
#
##nlp(data['body'])
##%%
### used mata fields 
### 'language_code' 'region_codes''snippet''title' 'body' 'an' 'publication_date'
#    
#meta_list = ['language_code','region_codes','snippet','title','an','publication_date'] #,'body'
#
#for m in meta_list:
#    print('{}: {}'.format(m,data[m]))
#    
##%%
### test how bing is the file 


