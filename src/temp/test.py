import pickle
import os 
import pandas as pd
from stream import FileStreamer_fast as FileStreamer
#%%
class args_class(object):
    def __init__(self, in_dir,out_dir,period='crisis',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.period=period
        self.verbose = verbose

args = args_class('../cleaned_small','../data/doc_meta', verbose = False)

#%%
streamer = FileStreamer(args.in_dir, language='en',verbose=True)

#%%
files = list(streamer)

#%%
files2 = streamer.multi_process_files()