#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:43:46 2019

@author: chuang
"""

import sys
sys.path.insert(0,'./libs')
import config as config
import os 
import pandas as pd
#%%

file = os.path.join(config.DOC_META,'doc_details_crisis_aug.pkl')
df = pd.read_pickle(file)
#%%

