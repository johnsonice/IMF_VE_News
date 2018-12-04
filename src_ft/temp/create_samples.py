#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:52:06 2018

@author: chuang
"""


import os 
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config
from glob import glob
from shutil import copy2

#%%

out_folder_dir = '/data/News_data_raw/files_to_share/historical_processed'
files_path  = glob(config.JSON_LEMMA + "/*.json")[:100]

#%%
for f in files_path:
    copy2(f,out_folder_dir)