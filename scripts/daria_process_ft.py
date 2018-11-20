"""Minimum - Script to process FT data for OCR files: 1980-2010"""

from glob import glob
import argparse
import pickle
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
import spacy
nlp = spacy.load('en_core_web_lg')

import time

def spacy_oov(data):
    oov = 0
    #text = preprocess_html(text)
    doc = nlp(data)
    length = len(doc)
    for token in doc:
        if token.is_oov:
            oov+=1
    return length, oov

def check_length(bad_files):
    if len(bad_files)% 100 == 0:
        return True
    else:
        return False

def preprocess_html(s):
    # Attempt to normalize escaped characters in OCR files
    
    s = s.replace("&quot;",'"')
    s = s.replace("&apos;","'")
    s = s.replace("&lt;", "<")
    s = s.replace("&gt;", ">")
    # this has to be last:
    s = s.replace("&amp;", "&")
    return s

def preprocess(text):
    # Normalize capitalization
    text = text.lower()
    text = preprocess_html(text)
    # Normalize spacing
    text = re.sub("\s+", " ", text)
    # Normalize numbers (the fact that a number appears may be important, but not the actual number)
    text = re.sub("(\d+[,./]?)+", "<<NUMBER>>", text)
    return text


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
    startTime = time.time()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
    #parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
    #parser.add_argument("checkFile",default="./DNA_news_corpus_cleaned.zip")
    in_dir = 'E:\\data\\DUlybina\\FT_OCR\\json_zips\\'
    outdir = 'E:\\data\\DUlybina\\ftocr_v3\\'
    #args = parser.parse_args()
    
    #existing_files = dna_content(args.checkFile)
    #Files = [f for f in os.listdir(args.in_dir) if f.endswith('.zip')]
    Files = [f for f in os.listdir(in_dir) if f.endswith('.zip')]
    existing = [f.lower().replace(".json",".html.json") for f in os.listdir(outdir) if f.endswith('.json')]
    decoding_errors = []
    empty_files = []
    j = 0
    print("Processing...")
    for root in Files:
        #with zipfile.ZipFile(os.path.join(args.in_dir,root)) as z:
        with zipfile.ZipFile(os.path.join(in_dir,root)) as z:
            for name in z.namelist():
                if name.endswith(".json") and name[-29:].lower() not in existing:
                    data = z.open(name)
                    try:
                        obj = json.load(reader(data))
                        articles = obj['articles']
                        text_list = [x['paras'] for x in articles]
                        flat_text = " ".join([item for sublist in text_list for item in sublist]).replace('- \n','')
                        flat_text = preprocess(flat_text)
                        length, oov = spacy_oov(flat_text)
                        if length > 0:
                            rates = oov/length
                        else:
                            rates = 0
                        titles_list = " ".join([x['title'] for x in articles]).replace("article: title: ","")
                        an = articles[0]['source_html_file_name'].lower()
                        pub_date = articles[0]['issue']
                        processed = {
                            'an':an.replace('.html',''),
                            'pub_date':pub_date,
                            'body': flat_text,
                            'title':titles_list,
                            'total_tokens': str(length),
                            'oov':str(oov),
                            'rate':str(rates)
                            }
                        with open(os.path.join(outdir,an.replace('.html','.json')), 'w') as outfile:
                            json.dump(processed, outfile)
                        j += 1
                        if  j % 10000 == 0:
                            print("Processed {} docs".format(str(j)))
                            print("Last doc: {}".format(an))
                            print("From directory: {}".format(name)) 
                    except json.JSONDecodeError:
                        decoding_errors.append(name)
                        print("Decoding error: {}".format(name))
                        continue
                    except IndexError:
                        empty_files.append(name)
                        print("No length for article: {}".format(name))
                        continue
                else:
                    print("{} already exisits".format(name[-29:].lower()))
                    
                    
    with open('E:\\data\\DUlybina\\empty_ft_list.pkl', 'wb') as f:
        pickle.dump(empty_files, f)
    with open('E:\\data\\DUlybina\\decoding_error_ft_list.pkl', 'wb') as f:
        pickle.dump(decoding_errors, f)
    
                    
    print("Total files written: {}".format(str(j)))
    print("Decoding errors: {}\n\n".format(len(decoding_errors)))
    print(decoding_errors)
    print("------------------------------------------")

    endTime = time.time()
    print("Processed in {} secs".format(str(endTime-startTime)))

    #making_archive(args.out_dir)
    
