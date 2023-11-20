# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:08:27 2023

@author: CHuang
"""
import os,json,re,time
import functools
import logging 
import datetime 
from functools import wraps
now = datetime.datetime.now()
#name = os.getlogin()
USER = 'chuang' #name.upper()
file_path = f"log/{USER}/{datetime.date.today()}"
os.makedirs(file_path,exist_ok=True)
filename = f"{file_path}/Exp-{now.hour}:{now.minute}.log"
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=filename,
    filemode="w",
    format=fmt
    )

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

def load_json(f_path):
    with open(f_path) as f:
        data = json.load(f)
    
    return data

def load_jsonl(fn):
    result = []
    with open(fn, 'r') as f:
        for line in f:
            data = json.loads(line)
            result.append(data)
    return result 

def to_jsonl(fn,data,mode='w'):
    with open(fn, mode) as outfile:
        if isinstance(data,list):
            for entry in data:
                json.dump(entry, outfile)
                outfile.write('\n')
        else:
            json.dump(data, outfile)
            outfile.write('\n')

def list_difference_left(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    return list(set1 - set2)

def retry(attempts=3, delay=1,raise_error=True):
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
                        return None
        return wrapper
    return decorator

def exception_handler(error_msg='error handleing triggered',
                        error_return=None,
                        attempts=3,delay=1):
    '''
    follow: https://stackoverflow.com/questions/30904486/python-wrapper-function-taking-arguments-inside-decorator
    '''
    def outter_func(func):
        @wraps(func)
        def inner_function(*args, **kwargs):
            for i in range(attempts):
                try:
                    res = func(*args, **kwargs)
                    return res
                except Exception as e:
                    if i < attempts - 1:
                        print(f"Function failed with error {e}. Retrying after {delay} seconds...")
                        time.sleep(delay)
                    else:
                        custom_msg = kwargs.get('error_msg', None)
                        if custom_msg:
                            logging.error(custom_msg)
                        else:
                            logging.error(str(e))
                        res = error_return
                        return res 
        return inner_function
    
    return outter_func

class Args:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                setattr(self, key, value)




