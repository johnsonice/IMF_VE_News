"""Minimum - Script to process FT data for OCR files: 1980-2010"""

from glob import glob
import argparse

import pandas as pd
import numpy as np
import gc
gc.collect()
import zipfile
import os, sys, re, csv, json
from pprint import pprint
import codecs
import itertools,string
reader = codecs.getreader("utf-8")

import time


def preprocess_html(s):
    # Attempt to normalize escaped characters in OCR files
    
    #s = s.replace("&quot;",'"')
    #s = s.replace("&apos;","'")
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    # this has to be last:
    s = s.replace("&amp;", "&")
    return s

def preprocess(text):
    # Normalize capitalization
    text = text.lower()
    # Normalize spacing
    text = re.sub("\s+", " ", text)
    # Normalize numbers (the fact that a number appears may be important, but not the actual number)
    text = re.sub("(\d+[,./]?)+", "<<NUMBER>>", text)
    return text

def create_dictionary(i,article):
    an = article['source_html_file_name'].lower()
    title = article['title']
    pub_date = article['issue']
    assert isinstance(title, str), "title variable should be string"
    text = " ".join([x for x in article['paras']])
    assert isinstance(text, str), "transformed text variable should be string"
    text = preprocess(text)
    identifier = an[:-5]+str(i)
    
    source = "Financial Times" 
    adict = {
        'an': identifier,
        'language_code': 'en', 
        'body': text,
        'publication_date': pub_date,
        'source_name':source,
        'title': title,
        'region_codes':'NA',
        #'snippet':text[0:141]
        }
    return adict,identifier

def save_to_json(adict,identifier,outdir):
    jsonData = json.dumps(adict)
    json_path = os.path.join(outdir,identifier+".json")
    with open(json_path, 'w') as f:
        json.dump(jsonData, f)
    return
      
def making_archive(outdir):
    archive_start = time.time()
    one_up = os.path.abspath(os.path.join(outdir,".."))
    make_archive(
        'ft_ocr.zip',
        'zip',           # the archive format - or tar, bztar, gztar
        root_dir=one_up,   # root for archive - current working dir if None
        base_dir=outdir)   # start archiving from here - cwd if None too
    archive_end = time.time()
    print("Archived in {} secs".format(str(archive_end - archive_start)))
    return


def dna_content(checkdir):
    all_names = []
    with zipfile.ZipFile(checkdir) as z:
        for info in z.namelist():
            all_names.append(info[8:])
    return all_names

##############################################################

if __name__ == '__main__':
    #startTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
    parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
    parser.add_argument("checkFile",default="./DNA_news_corpus_cleaned.zip")
    args = parser.parse_args()
    
    existing_files = dna_content(args.checkFile)
    Files = [f for f in os.listdir(args.in_dir) if f.endswith('.zip')]

    decoding_errors = []
    j = 0
    print("Processing...")
    for root in Files:
        with zipfile.ZipFile(os.path.join(args.in_dir,root)) as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    data = z.open(name)
                    try:
                        obj = json.load(reader(data))
                        articles = obj['articles']
                        for i,article in enumerate(articles):
                            adict,ident = create_dictionary(i,article)
                            if ident not in existing_files:
                                save_to_json(adict,ident,args.out_dir)
                                j += 1
                                if  j % 10000 == 0:
                                    print("Processed {} docs".format(str(j)))
                                    print("Last doc: {}".format(ident))
                                    print("From directory: {}".format(name))
                            else:
                                print('!!! WARNING: Duplicate found!')
                                print(ident)
                                save_to_json(adict,ident+"_"+str(j),args.out_dir)
                                j += 1
                    except json.JSONDecodeError:
                        decoding_errors.append(name)
                        print("Decoding error")
                        continue
                    
    print("Total files written: {}".format(str(j)))
    print("Decoding errors: {}\n\n".format(len(decoding_errors)))
    print(decoding_errors)
    print("------------------------------------------")

    #endTime = time.time()
    #print("Processed in {} secs".format(str(endTime-startTime)))

    #making_archive(args.out_dir)
