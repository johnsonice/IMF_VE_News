#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:33:46 2019

@author: chuang
"""

import os 
import sys
import datetime as dt
## some inference utility functions 

def maybe_create(f_path):
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        print('Generate folder : {}'.format(f_path))
    return None

def get_current_date():
    res = dt.datetime.strftime(dt.datetime.today(),"%Y-%m-%d")
    return res

def get_current_month():
    res = dt.datetime.strftime(dt.datetime.today(),"%Y-%m")
    return res

