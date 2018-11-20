"""Minimum - Script to process FT data for OCR files: 1980-2010"""

#from glob import glob
#import argparse
import pickle
#import pandas as pd
#import numpy as np
import gc
gc.collect()
import zipfile
import os, sys, re, csv, json
#from pprint import pprint
import codecs
#import itertools,string
reader = codecs.getreader("utf-8")
#import spacy
#nlp = spacy.load('en_core_web_lg')
from functools import partial
from multiprocessing import Pool 
import time
import datetime
import html
from langdetect import detect ## for detecting language

#%%

def preprocess_html(s):
    # Attempt to normalize escaped characters in OCR files
    s = s.replace("&#x00ac; \n","")
#    s = s.replace("l&#x00ab; ","<<")
#    s = s.replace("&#x00ab;i", ">>")             
#    s = s.replace("&quot;",'"')
#    s = s.replace("&apos;","'")
#    s = s.replace("&lt;", "<")
#    s = s.replace("&gt;", ">")
#    s = s.replace("\'", "'")
#    # this has to be last:
#    s = s.replace("&amp;", "&")
    
    return html.unescape(s) ## one function to convert all html code to unicode 

def preprocess(text):
    # Normalize capitalization
    text = text.lower()
    text = preprocess_html(text)
    # Normalize spacing
    text = re.sub("\s+", " ", text)
    # Normalize numbers (the fact that a number appears may be important, but not the actual number)
    #text = re.sub("(\d+[,./]?)+", "<<NUMBER>>", text)
    return text

def dna_content(checkdir):
    all_names = []
    with zipfile.ZipFile(checkdir) as z:
        for info in z.namelist():
            all_names.append(info[8:])
    return all_names

def convert_date(issue_date):
    ## soemthing like ' February 21, 1998, page: 0038'
    clean_date = issue_date.replace('issue: date:','').split(', page:')[0].strip()
    format_str = '%B %d, %Y'
    date_obj = datetime.datetime.strptime(clean_date,format_str)
    transformed_str = date_obj.strftime("%Y-%m-%d")
    
    return transformed_str
    
def get_all_files(folder_path,end='.json'):
    for path,subdirs,files in os.walk(folder_path):
        for name in files:
            if name.endswith(end):
                yield os.path.join(path,name)

def transform_article(a,idx,lang_check=False):
    
    flat_text = " ".join(a['paras']).replace('- \n','')
    flat_text = preprocess(flat_text)
    snippet = " ".join(a['paras'][:2]).replace('- \n','')
    snippet = preprocess(snippet)
    
    if lang_check and len(flat_text)>10:  ## make sure there are contents in the file 
        lang = detect(flat_text)
    else:
        lang="NA"
    title = a['title'].replace("article: title: ","")
    an = a['source_html_file_name'].lower().replace('.html','')+"-{}".format(str(idx))
    pub_date = convert_date(a['issue'])
    processed = {
        'an':an,
        'language_code': lang,
        'publication_date':pub_date,
        'body': flat_text,
        'title':title,
        'region_codes':'NA',
        'snippet':snippet
        }
    
    return processed

def transform_dump(f_path,out_dir):
    try:
        with open(f_path,'r') as f:
            obj = json.load(f)
        ## dict_keys(['issue', 'source_html_file_name', 'description', 'articles'])
        articles = obj['articles']
        for idx,a in enumerate(articles):
            res = transform_article(a,idx,True)
            with open(os.path.join(out_dir,res['an']+'.json'), 'w') as outfile:
                json.dump(res, outfile)
    except json.JSONDecodeError:
        print("Decoding error: {}".format(f_path))
        return ('Decoding error',f_path)
    except IndexError:
        print("No length for article: {}".format(f_path))
        return ('empty file error',f_path)
    except:
        print("Other error: {}".format(f_path))
        return ('Other error',f_path)
    
    return None

def _chunks(l,n):
    """yield successive n-sized chunks form l."""
    for i in range(0,len(l),n):
        yield l[i:i+n]

def single_process_files(files,out_dir,print_iter=5000):
    res_list = list()
    for f_idx,f_path in enumerate(files):
        res = transform_dump(f_path,out_dir)
        res_list.append(res)
    
        if  f_idx % print_iter == 0:
            print("Processed {} docs".format(str(f_idx)))
            print("From directory: {}".format(f_path)) 
    
    return res_list

def multi_process_files(files,out_dir,workers,chunksize=1000):
    print("Start multi processing in {} cores ....".format(workers))
    batch_size = workers*chunksize*5
    batches = list(_chunks(files,batch_size))
    p=Pool(workers)
    partial_process = partial(transform_dump,out_dir = out_dir)
    ## process documents by batches 
    res_list = list()
    for i in range(len(batches)):
        print('Processing {} - {} files ...'.format(i*batch_size,(i+1)*batch_size))
        res = p.map(partial_process,batches[i],chunksize=chunksize)
        res_list.extend(res)
    
    p.close()
    p.join()    
    print('\nfinish')
    return res_list
#%%
if __name__ == "__main__":
    ## global variables
    in_dir = '/data/News_data_raw/Financial_Times/all_archieve_ocr'
    out_dir = '/data/News_data_raw/FT_json_historical'
    out_log = '/data/News_data_raw/FT_log'
    workers = 30
    
    ## set timer
    startTime = time.time()
    ## process files
    files = list(get_all_files(in_dir,end ='.json'))
    print('Total number of documents to process: {}'.format(len(files)))
    ## multiprocess or single process
    if workers > 1:
        res = multi_process_files(files,out_dir,workers=workers,chunksize=1000)
    else:
        res = single_process_files(files,out_dir,print_iter=5000)
        
    ## dump log file
    res = [r for r in res if r is not None]
    with open(os.path.join(out_log,'log_historical.pkl'), 'wb') as f:
        pickle.dump(res, f)
    
    print("Total files written: {}".format(len(files)))
    print("errors: {}\n".format(len(res)))
    #print(res)
    print("------------------------------------------")
    endTime = time.time()
    print("Processed in {} secs".format(time.strftime('%H:%M:%S',time.gmtime(endTime-startTime))))


#%%
### run on test 
#out_dir = '/data/News_data_raw/Financial_Times/all_archieve_ocr'
#files = list(get_all_files(in_dir,end ='.json'))
##%%
#f_path = files[1]
#with open(f_path,'r') as f:
#    obj = json.load(f)
#
#obj['articles'][2]



