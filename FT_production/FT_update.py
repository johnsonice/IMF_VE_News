#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:48:26 2019

@author: chuang
"""

## FT updator 
import sys
import os
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
sys.path.insert(0,os.path.join(cwd,'./inference_libs'))
sys.path.insert(0,os.path.join(cwd,'./inference_libs/libs'))

import infer_config as config
from data_process import Data_processor
from meta_generator import Meta_processor 
from country_freq_generator import Freq_generator 
from time_series_generator import TS_generator, get_ts_args
from result_merger import data_updator,get_dm_args
from country_data_aggregator import export_tableau_data,long_to_wide
from data_cleaner import clean_folder,backup_folder,backup_file
from ft_api import download_docs,ft_api_args


from stream import MetaStreamer_fast as MetaStreamer
from crisis_points import crisis_points,country_dict
import datetime as dt
from mp_utils import Mp
from infer_utils import get_current_date,get_current_month


def download_current_files():
    ## step 0 dowloading current month of data 
    api_args = ft_api_args()
    api_args.cred_path = os.path.join('/data/News_data_raw/Production','credentials_ft.txt')
    
    
    ###############################################################################
    ## if you want to change some input on the fly, you can do it here 
    ## note we can only go back for 3 month 
#    api_args.date_since = '2020-07-01'
#    api_args.data_dir = os.path.join(config.JSON_RAW,"{}".format('2020-07-08'))
    ###############################################################################
    
    print(api_args)
    download_docs(api_args)
    
def process_raw_data():
    ## step 1 process raw data 
    ## copy current month data to processing folder 
    
    current_month = get_current_month()
    ###############################################################################
    ##### if you changed download folder name, you need to change here too 
#    current_month = '2020-07-08'
#    current_month = '2019_0408'
    ##############################################################################
    
    current_month_data_folder = os.path.join(config.JSON_RAW,current_month)
    assert os.path.exists(
            current_month_data_folder
                          ),"Current month data folder do not exist; please make sure you downloaded current data"
    backup_folder(current_month_data_folder,os.path.join(config.HISTORICAL_INPUT,current_month),overwrite=True)
    
    ## step 2 process raw data 
    dp = Data_processor()
    dp.pre_process_files(config.JSON_RAW,config.JSON_LEMMA,config.HISTORICAL_INPUT,end_with='.json',n_worker=15)
    

def generate_meta_file(keep_current=True):
    ## step 3 generate metadata file 
    current_month = get_current_month()
    ## initiate meta generator 
    mg = Meta_processor(MetaStreamer,crisis_points)
        ## create meta
    df_meta = mg.generate_meta(config.JSON_LEMMA,config.DOC_META)
        ## add a temp filter on the fly 
        
    
    #############################################################################################
    ##current_month = dt.datetime.strftime(dt.datetime.today(),"%Y-%m")
    ##current_month = '2019-04' # you can change here if you don't just want current month
#    keep_current = False  ## set to false if you don't wnat to do any filtering
#    df_meta = df_meta[df_meta['month']>= '2020-07'] ## or put in a specific month e.g: '2019-04'
    #################################################################################################
    
    if keep_current:
        print('Filter docs to keep only current month: {}'.format(current_month))
        df_meta = df_meta[df_meta['month']==current_month] ## or put in a specific month e.g: '2019-04'
    
    ## save meta file to location
    df_meta.to_pickle(config.DOC_META_FILE)
        ## export summary statistics
    mg.create_summary(df_meta,meta_root=config.DOC_META)
    mg.export_country_stats(df_meta,country_dict,config.DOC_META)
    
def generate_country_bow():
    ## step 4 generate country specific bag of words representation
    # obtain freqs
    print(config.COUNTRY_FREQ_PERIOD)
    Fg = Freq_generator(config.DOC_META_FILE)
    Fg.get_country_freqs(config.countries, config.JSON_LEMMA,config.COUNTRY_FREQ_PERIOD, config.FREQUENCY,config.PHRASER)

def generate_country_time_series():
    ## step 5 generate time series based on new data 
    ts_args = get_ts_args(config)
    ts_generator = TS_generator(ts_args)
    mp = Mp(config.countries,ts_generator.export_country_ts)
    res = mp.multi_process_files(chunk_size=1,workers=15)

def merge_with_historical():
    ## step 6 merge new data with historical data 
    dm_args = get_dm_args(config)
    du = data_updator(dm_args)
    du.export_all_updated_data()

def create_data_for_tableau():
    ## step 7 generate tableau input file 
    res_df = export_tableau_data(config.CURRENT_TS_PS,os.path.join(config.OUTPUT_FOLDER,'data_backup'))
    
    return res_df

def export_data_wide(df,local_back_up =True):
    wide_df = long_to_wide(df)
    out_path = os.path.join(config.OUTPUT_FOLDER,'data_backup','country_data_wide_{}.csv'.format(get_current_date()))
    try:
        if local_back_up:
            wide_df.to_csv(os.path.join(config.LOCAL_BACKUP,'country_data_wide_{}.csv'.format(get_current_date())))
        wide_df.to_csv(out_path,index=False)
    except:
        print('Warning !!! output folder access denied')
        return wide_df
    return wide_df
    
def backup_and_clean_up():
    ## step 8 
    ## step clean up and back up folder 
        # copy current as historical
    backup_folder(config.CURRENT_TS_PS,config.HISTORICAL_TS_PS,overwrite=True)
        # copy historical to a backup 

    backup_folder(config.HISTORICAL_TS_PS,os.path.join(config.BACKUP_TS_PS,get_current_date()),overwrite=True)

        # clean up current folder 
    folder_clean = [config.CURRENT_TS_PS,config.JSON_LEMMA,config.JSON_RAW,config.DOC_META,config.FREQUENCY]
        
    ## step 9 
    ## move file to production tableau path
    long_data_path = os.path.join(config.OUTPUT_FOLDER,'data_backup','country_data_long_{}.csv'.format(get_current_date()))
    wide_data_path = os.path.join(config.OUTPUT_FOLDER,'data_backup','country_data_wide_{}.csv'.format(get_current_date()))
    tableau_production_data_path = os.path.join(config.OUTPUT_FOLDER,'dashboard','country_data_long.csv')
    data_share_path = os.path.join(config.OUTPUT_FOLDER,'data_share','country_data_wide.csv')
    
    backup_file(long_data_path,tableau_production_data_path,overwrite=True)
    backup_file(wide_data_path,data_share_path,overwrite=True)
    
    for f in folder_clean:
        clean_folder(f)
        
#%%
if __name__ == "__main__":
    print('\n......runing......\n')

    download_current_files()
    process_raw_data()
    #%%
    generate_meta_file()
    #%%
    generate_country_bow()
    generate_country_time_series()
    merge_with_historical()

    long_df = create_data_for_tableau()
    wide_df = export_data_wide(long_df)
    backup_and_clean_up()
    
    print('Update success')




