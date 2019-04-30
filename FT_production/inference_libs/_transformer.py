"""Minimum - Script to process FT data for OCR files: 1980-2010"""
import pickle
import gc
gc.collect()
import zipfile
from bs4 import BeautifulSoup as bs
import os, sys, re, csv, json
import codecs
reader = codecs.getreader("utf-8")
#import spacy
#nlp = spacy.load('en_core_web_lg')
from functools import partial
from multiprocessing import Pool 
import time
import datetime
import pandas as pd
import html
from langdetect import detect ## for detecting language

import spacy
#from spacy.lang.en.stop_words import STOP_WORDS as stops
nlp = spacy.load("en_core_web_lg",disable=['tagger','ner','parser','textcat'])
from spacy.symbols import ORTH, LEMMA, POS, TAG
special_case = [{ORTH:u'__NUMBER__',LEMMA:u'__NUMBER__'}]
nlp.tokenizer.add_special_case(u'__NUMBER__',special_case)

#%%

def preprocess(text,lemma=True):
    try:
        #preprocess html content
        soup = bs(text, 'lxml') #'html.parser'
        text = soup.text
        # Normalize capitalization
        text = text.lower()
        # Normalize spacing
        text = re.sub("\s+", " ", text)
        text = re.sub("\'+", " ", text)
        
        # Normalize numbers (the fact that a number appears may be important, but not the actual number)
        #text = re.sub("(\d+[,./]?)+", "__NUMBER__", text)
        text = re.sub("([',./]?\d+[',./]?)+", " __NUMBER__ ", text)
        
        # lemmentize 
        if lemma:
            toks = nlp(text)
            toks = [tok.lemma_ for tok in toks if not tok.is_space]
            text = ' '.join(toks)
            
        return text

    except Exception:
        print('no text in article')
        return False        


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

def transform_article(article,locations=None,lang_check=False):
    an = article['id'].lower()
    title = article['title']
    pub_date = article['publishedDate'].split('T')[0]
    identifier = "ft-"+an.split('/')[-1] ## get only the id 
    assert isinstance(title, str), "title variable should be string"
    
    ## this part can eror out, it will be catched in log file 
    text = article['bodyXML'] ## some documents don't have body xml 
    text = preprocess(text)
    snippet = " ".join(text.split()[:50])
    
    if lang_check and len(text)>10:
        lang = detect(text)
    else:
        lang = "NA"
    #print(lang)
    
    #regs = locations.get(article['id'])
    source = "Financial Times"
    adict = {
        'an': identifier,
        'language_code': lang,
        'body': text,
        'publication_date': pub_date,
        'source_name':source,
        'title': title,
        'region_codes':'NA',
        'snippet': snippet
        }
    return adict  

def transform_dump(f_path,out_dir):
    try:
        with open(f_path,'r') as f:
            obj = json.load(f)
        res = transform_article(obj,lang_check=True)
        ## make sure body is not empty
        if len(res['body']) >0:
            with open(os.path.join(out_dir,res['an']+'.json'), 'w') as outfile:
                json.dump(res, outfile)
    except json.JSONDecodeError:
        print("Decoding error: {}".format(f_path))
        return ('Decoding error',f_path)
    except IndexError:
        print("No length for article: {}".format(f_path))
        return ('empty file error',f_path)
    except KeyError:
        #print('Key error, no [bodyXML]: {}'.format(f_path))
        return ('Key error',f_path)
    except:
        print('Other error: {}'.format(f_path))
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
    in_dir = '/data/News_data_raw/Production/data/raw_input_current_month/'
    #content_dir = '/data/News_data_raw/Financial_Times/FT-archive-concepts'
    out_dir = '/data/News_data_raw/Production/data/input_processed_current_month/'
    out_log = '/data/News_data_raw/Production/data/raw_input_current_month/'
    workers = 25
    
    ## set timer
    startTime = time.time()
    ## process files
    files = list(get_all_files(in_dir,end ='.json'))

    ## read meta data sheet info : it is not complete, not very useful 
    #topics, locations = get_concepts(content_dir)
    
    print('Total number of documents to process: {}'.format(len(files)))
    ## multiprocess or single process
    if workers > 1:
        res = multi_process_files(files,out_dir,workers=workers,chunksize=2000)
    else:
        res = single_process_files(files,out_dir,print_iter=5000)
        
    ## dump log file
    res = [r for r in res if r is not None]
    with open(os.path.join(out_log,'log_18m6_19m4.pkl'), 'wb') as f:
        pickle.dump(res, f)
    
    print("Total files written: {}".format(len(files)))
    print("errors: {}\n".format(len(res)))
    #print(res)
    print("------------------------------------------")
    endTime = time.time()
    print("Processed in {} secs".format(time.strftime('%H:%M:%S',time.gmtime(endTime-startTime))))

#%%
#test_file = '/data/News_data_raw/Financial_Times/all_current/FT-archive-2016/74e5faed-0041-3e0b-8672-7d967ecacae1_2016-07-07.json'
#with open(test_file,'r') as f:
#    obj = json.load(f)
#print(obj.keys())



