#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 08:54:16 2019

@author: chuang
"""


import sys,os
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config
import pandas as pd
from datetime import datetime as dt

#%%
config.FREQUENCY = '/data/News_data_raw/FT_WD/frequency/csv'
#%%
test_csv = os.path.join(config.FREQUENCY,'test_argentina_month_word_freqs.csv')
#%%

df = pd.read_csv(test_csv,index_col=0)
df.head()
#%%
test_pkl = os.path.join('/data/News_data_raw/FT_WD/frequency/argentina_quarter_word_freqs.pkl')
df_p = pd.read_pickle(test_pkl)


