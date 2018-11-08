"""Minimum - Script to process FT data for nrewer (not OCR) FT files """
from glob import glob
import argparse

import pandas as pd
import numpy as np
import gc
gc.collect()
import zipfile
import os, sys, re, csv, json
from pprint import pprint
from bs4 import BeautifulSoup as bs
import codecs
import itertools,string
#from datetime import datetime
reader = codecs.getreader("utf-8")
from shutil import make_archive
import time


def get_concepts(indir):
    concept_path = os.path.join(indir,"FT-archive-concepts")
    for file in os.listdir(concept_path):
        if file == 'Topic.csv':
            df = pd.read_csv(os.path.join(concept_path,file))
            df = df[['id','prefLabel']]
            topics = df.set_index('id')['prefLabel'].to_dict()
        elif file =='Location.csv':
            df = pd.read_csv(os.path.join(concept_path,file))
            df = df[['id','prefLabel']]
            locations = df.set_index('id')['prefLabel'].to_dict()
    return topics, locations

def save_to_json(adict,name,outdir):
    jsonData = json.dumps(adict)
    json_path = os.path.join(outdir,name)
    with open(json_path, 'w') as f:
        json.dump(jsonData, f)
    return

def preprocess(text):
    #preprocess html content
    soup = bs(text, 'lxml') #'html.parser'
    text = soup.text
    # Normalize capitalization
    text = text.lower()
    # Normalize spacing
    text = re.sub("\s+", " ", text)
    # Normalize numbers (the fact that a number appears may be important, but not the actual number)
    text = re.sub("(\d+[,./]?)+", "<<NUMBER>>", text)
    return text

def create_dictionary(article,locations):
    an = article['id'].lower()
    title = article['title']
    pub_date = article['publishedDate']
    assert isinstance(title, str), "title variable should be string"
    try:
        text = article['bodyXML']
        text = preprocess(text)
    except KeyError:
        text = ''
        print("!!! WARNING: No text in: {}".format(an))
    identifier = "ft"+an[24:]
    #regs = locations.get(article['id'])
    source = "Financial Times"
    adict = {
        'an': identifier,
        'language_code': 'en',
        'body': text,
        'publication_date': pub_date,
        'source_name':source,
        'title': title,
        'region_codes':'NA'
        #'snippet':text[0:141]
        }
    return adict,identifier  

def making_archive(outdir):
    archive_start = time.time()
    one_up = os.path.abspath(os.path.join(outdir,".."))
    make_archive(
        'ft_newer.zip',
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
    topics, locations = get_concepts(args.in_dir)

    j = 0
    print("Processing...")
    for root in Files:
        with zipfile.ZipFile(os.path.join(args.in_dir,root)) as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    data = z.open(name)
                    obj = json.load(reader(data))
                    adict, identifier = create_dictionary(obj,locations)
                    #print("Assigned identifier: {}".format(identifier))
                    if name not in existing_files:
                        save_to_json(adict,name,args.out_dir)
                        j += 1
                        if  j % 10000 == 0:
                            print("Processed {} docs".format(str(j)))
                            print("Last doc: {}, from dir: {}".format(name,root))
                    else:
                        print('!!! WARNING: Duplicate found!')
                        print(name)
                        identifier = identifier + "_"+str(j)+'.json'
                        save_to_json(adict,identifier,args.out_dir)
                        j += 1
                        
                    
    #endTime = time.time()
    #print("Processed in {} secs".format(str(endTime-startTime)))
    #making_archive(args.out_dir)
    

