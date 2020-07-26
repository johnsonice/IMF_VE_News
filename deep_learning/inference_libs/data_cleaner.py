#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:11:54 2019

@author: chuang
"""

## data cleaner 
import sys
import os
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))
import shutil
import infer_config as config
from infer_utils import get_current_date
#%%


def clean_folder(folder_path):
    for the_file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def backup_folder(old,new,overwrite=False):
    if overwrite:
        if os.path.exists(new):
            shutil.rmtree(new)
            print('over write existing folder')
    shutil.copytree(old,new)

def backup_file(old,new,overwrite=False):
    if overwrite:
        if os.path.exists(new):
            os.remove(new)
            print('over write existing file')
    shutil.copyfile(old,new)