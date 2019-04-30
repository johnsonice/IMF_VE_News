#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:03:07 2019

@author: chuang
"""

import sys
import os
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
sys.path.insert(0,os.path.join(cwd,'./libs'))

import time
import _transformer as transformer
import pickle

import argparse
#%%
#from gensim.models.keyedvectors import KeyedVectors
#from crisis_points import crisis_points,ll_crisis_points
#from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats
#from frequency_utils import rolling_z_score, aggregate_freq, signif_change
#import pandas as pd
#import numpy as np
#from mp_utils import Mp
#import os
#import config

#%%

class Data_processor (object):
    """
    data processor for transforming raw ft data;
    basic clean up and text normalization
    """
    
    def __init__(self,transformer):
        self.transformer = transformer
    
    @staticmethod
    def _maybe_create(f_path):
        if not os.path.exists(f_path):
            os.makedirs(f_path)
            print('Generate folder : {}'.format(f_path))
        return None
    
    def pre_process_files(self,in_dir,out_dir,log_dir,end_with='.json',n_worker=1,verbose=True):
        #3 check filder exist 
        self._maybe_create(out_dir)
        self._maybe_create(log_dir)
        ## set timer
        startTime = time.time()
        ## process files
        files = list(self.transformer.get_all_files(in_dir,end =end_with))
        if verbose:
            print('Total number of documents to process: {}'.format(len(files)))
        ## multiprocess or single process
        if n_worker > 1:
            res = dp.transformer.multi_process_files(files,out_dir,workers=n_worker,chunksize=2000)
        else:
            res = dp.transformer.single_process_files(files,out_dir,print_iter=5000)
            
        ## dump log file
        res = [r for r in res if r is not None]
        with open(os.path.join(log_dir,'log.pkl'), 'wb') as f:
            pickle.dump(res, f)
        
        if verbose:
            print("Total files written: {}".format(len(files)))
            print("errors: {}\n".format(len(res)))
            #print(res)
            print("------------------------------------------")
            endTime = time.time()
            print("Processed in {} secs".format(time.strftime('%H:%M:%S',time.gmtime(endTime-startTime))))    
                
        return res
#%%
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', 
                        default='/data/News_data_raw/Production/data/raw_input_current_month/')
    parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', 
                        default='/data/News_data_raw/Production/data/input_processed_current_month/')
    parser.add_argument('-ol', '--log_dir', action='store', dest='log_dir', default='/data/News_data_raw/Production/data/raw_input_current_month/')
    parser.add_argument('-nw', '--n_workers', action='store', dest='n_workers', type=int,default=1)
    args = parser.parse_args()
    
    ## initiate object 
    dp = Data_processor(transformer)
    ## preprocess all input files 
    dp.pre_process_files(args.in_dir,args.out_dir,args.log_dir,end_with='.json',n_worker=15)

