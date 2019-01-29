import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'../libs')
from stream import MetaStreamer_fast as MetaStreamer
import ujson as json
#%%
class args_class(object):
    def __init__(self, in_dir,out_dir,period='crisis',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.period=period
        self.verbose = verbose

args = args_class('../../data/processed_json','../../data/doc_meta', verbose = False)

#%%
streamer = MetaStreamer(args.in_dir, language='en',verbose=True)
files = streamer.input_files
#%%

with open(files[10], 'r', encoding="utf-8") as f:
    data = json.loads(f.read())
    
## used mata fields 
## 'language_code' 'region_codes''snippet''title' 'body' 'an' 'publication_date'
    
meta_list = ['language_code','region_codes','snippet','title','an','publication_date'] #,'body'

for m in meta_list:
    print('{}: {}'.format(m,data[m]))
    
