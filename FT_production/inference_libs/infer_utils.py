#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:33:46 2019

@author: chuang
"""

import os 
import sys

## some inference utility functions 

def maybe_create(f_path):
    if not os.path.exists(f_path):
        os.makedirs(f_path)
        print('Generate folder : {}'.format(f_path))
    return None
    

