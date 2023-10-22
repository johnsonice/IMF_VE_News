# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:08:27 2023

@author: CHuang
"""
import os
import json
import time
import functools

def get_all_files(directory, end_with=None,start_with=None,return_name=False):
    files = []

    # os.walk yields a 3-tuple for each directory it visits.
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))


    if end_with:
        files = [f for f in files if f.endswith(end_with)]
    
    if start_with:
        files = [f for f in files if os.path.basename(f).startswith(start_with)]
    
    if return_name:
        files = [os.path.basename(f) for f in files]
    
    return files


def read_json(f_p):
    with open(f_p, 'r',encoding='utf8') as f:
        data = json.load(f)
    
    return data 


def list_difference_left(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    return list(set1 - set2)

def retry(attempts=3, delay=1,raise_error=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i < attempts - 1:  # i is zero indexed
                        print(f"Function failed with error {e}. Retrying after {delay} seconds...")
                        time.sleep(delay)
                    else:
                        print("Function failed after several attempts. Raising the exception...")
                        if raise_error:
                            raise
        return wrapper
    return decorator



#%%







