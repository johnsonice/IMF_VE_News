# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:03:50 2023

@author: CHuang
"""

import pickle 
import pandas as pd 
import json,os

#%%
def load_pkl(fp):
    # Open the PKL file in read mode
    with open(fp, 'rb') as file:
        # Load the contents of the file
        data = pickle.load(file)
    return data 

def get_all_files(directory, end_with=None):
    files = []

    # os.walk yields a 3-tuple for each directory it visits.
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if end_with:
                if filename.endswith(end_with):  # check if the file is a JSON file
                    files.append(os.path.join(dirpath, filename))
            files.append(os.path.join(dirpath, filename))

    return files

#%%
if __name__ =="__main__":
    # Print the loaded data
    
    data_folder = r'Q:\DATA\SPRAI\SPRAI_Projects\News_Search\sandbox\data'
    files = get_all_files(data_folder, end_with=None)
    
    #%%
    output_f = r'Q:\DATA\SPRAI\SPRAI_Projects\News_Search\sandbox\data\reuters_2020-01-01_2020-03-01.json'
    with open(output_f, 'r',encoding='utf8') as f:
        data = json.load(f)
        