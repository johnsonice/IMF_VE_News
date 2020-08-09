#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 18:24:09 2020

@author: chuang
"""

import pandas as pd 
import numpy as np
pd.set_option('display.max_columns', None)
from dateutil.parser import parse
import config
import os
from crisis_points import crisis_points 
from dateutil.parser import parse

#%%

def get_crisis_dates_range(cisis_dict):
    starts = []
    ends = []
    for country,v in cisis_dict.items():
        starts += v['starts']
        ends += v['peaks']
    start = min(starts)
    end = max(ends)
    
    return start,end


#%%8
if __name__ == "__main__":
  
    crisis_date_path = os.path.join(config.CRISIS_DATES,'criris_dates.pkl')
    df_crisis_dates = pd.read_pickle(crisis_date_path)
    #%%
    start,end = get_crisis_dates_range(crisis_points)
    
    #%%
    start_date = parse(start)
    end_date = parse(start)
    #%%
    time_window = pd.PeriodIndex(pd.date_range(start,end,freq='m'),freq='m')
    #time_window
    
    #%%
    