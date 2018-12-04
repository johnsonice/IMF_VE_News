# -*- coding: utf-8 -*-

import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'../libs')
from stream import MetaStreamer_fast as MetaStreamer
import ujson as json

#%%
file_path = '/data/News_data_raw/FT_WD/doc_meta/doc_details_crisis.pkl'
df = pd.read_pickle(file_path)