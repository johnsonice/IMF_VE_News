#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:54:06 2019

@author: chuang
"""
import sys
import os
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
sys.path.insert(0,os.path.join(cwd,'./libs'))
#import config
import argparse
import pandas as pd
from datetime import datetime as dt
#import ujson as json
#from glob import glob
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from crisis_points import crisis_points,country_dict
from frequency_utils import list_crisis_docs

from stream import MetaStreamer_fast as MetaStreamer
from mp_utils import Mp
from nltk.tokenize import word_tokenize

#%%

class Meta_processor(object):
    
    def __init__(self,streamer,crisis_points):
        self.streamer = MetaStreamer
        self.crisis_points = crisis_points
        print('Meta generator initialized...')
        
        
    ## multi processed json input
    @staticmethod
    def time_index( docs, lang=None, verbose=False,date_format='ft'):
        doc_details = {}
        tot = len(docs)
        print('Convert dates....')
        for i, doc in enumerate(docs):
            if verbose:
                print('\r{} of {} processed'.format(i, tot), end='',flush=True)
            try:
                if date_format.strip().lower() == 'ft':
                    date = pd.to_datetime(dt.strptime(doc['publication_date'],'%Y-%m-%d'))
                else:
                    date = pd.to_datetime(dt.fromtimestamp(doc['publication_date'] / 1e3))
                doc_details[doc['an']] = {'date': date}
            except Exception as e:
                print(doc['an'] + ': ' + str(e))
                
        data = pd.DataFrame(doc_details).T
        return data
    
    @staticmethod
    def period_info(doc_deets):
        dates = pd.DatetimeIndex(doc_deets['date'])
        doc_deets['week'] = dates.to_period('W')
        doc_deets['month'] = dates.to_period('M')
        doc_deets['quarter'] = dates.to_period('Q')
        return doc_deets
    

    def label_crisis(self,data, path,verbose=False, period='crisis'):
        """
        path = meta data folder path
        """
        data['crisis'] = 0
        crisis = []
        for country in self.crisis_points.keys():
            if verbose:
                print("\nworking on {}...".format(country))
            crisis_docs = list_crisis_docs(country, path,doc_data=data, period=period)
            crisis_ids = [os.path.basename(doc).replace(".json", '') for doc in crisis_docs]
            crisis += crisis_ids
        data.loc[data.index.isin(crisis), 'crisis'] = 1
        return data
    @staticmethod
    def get_country_name(tokens,country_dict):
        for c,v in country_dict.items():
            rc = c if tokens and any([tok.lower() in tokens for tok in v]) else None
            if rc is not None:
                yield rc 
    
    def get_countries(self,article,country_dict=country_dict):
        snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
        title = word_tokenize(article['title'].lower()) if article['title'] else None
    
        if snip and title:
            title.extend(snip)
            cl = list(self.get_country_name(title,country_dict))
        elif title:
            cl = list(self.get_country_name(title,country_dict))
        elif snip:
            cl = list(self.get_country_name(snip,country_dict))
        else:
            cl = list()
            
        return article['an'],cl    
    
    def label_country(self,date_df):
        streamer = MetaStreamer(date_df['data_path'].tolist())
        news = streamer.multi_process_files(workers=30,chunk_size=5000)
        #country_meta = [(a['an'],get_countries(a,country_dict)) for a in news]
        mp = Mp(news,self.get_countries)
        country_meta = mp.multi_process_files(workers= 30,chunk_size=5000)
        index = [i[0] for i in country_meta]
        country_list = [i[1] for i in country_meta]
        del country_meta ## clear memory
        ds = pd.Series(country_list,name='country',index=index)
        date_df = date_df.join(ds) ## merge country meta
        date_df['country_n'] = date_df['country'].map(lambda x: len(x))
        
        return date_df
    
    
    @staticmethod
    def _maybe_create(f_path):
        if not os.path.exists(f_path):
            os.makedirs(f_path)
            print('Generate folder : {}'.format(f_path))
        return None
    
    
    def generate_meta(self,in_dir,out_dir=None,period='crisis',save=True,verbose=True):
        self._maybe_create(out_dir)
        streamer = MetaStreamer(in_dir, language='en',verbose=True)  
        date_df = self.time_index(streamer.multi_process_files(workers=25,chunk_size=5000), lang='en', verbose=False,date_format='FT')
        date_df = self.period_info(date_df)
        print('label crisis')
        date_df = self.label_crisis(date_df, path = in_dir,verbose=verbose, period=period)
        date_df['data_path'] = in_dir +'/'+date_df.index + '.json'
        print('see one example : \n',date_df['data_path'].iloc[0])
        date_df = self.label_country(date_df)
        
        
        if save and out_dir:
            date_df.to_pickle(os.path.join(out_dir, 'doc_details_{}.pkl'.format(period)))
        return date_df
    
    ## save mohtlhy figure
    @staticmethod
    def create_summary(df,meta_root=None,interval=1):
        """
        input meta df
        """
        agg_m= df[['date','month']].groupby('month').agg('count')
        x_ticker = agg_m.index[0::interval]
        ax = agg_m.plot(figsize=(16,6),title='News Articles Frequency',legend=False)
        plt.ylabel('Number of news articles')       ## you can also use plt functions 
        plt.xlabel('Time-M') 
        ax.set_xticks(x_ticker)                     ## set ticker
        ax.set_xticklabels(x_ticker,rotation=90)    ## set ticker labels
        ax.get_xaxis().set_visible(True)            ## some futrther control over axis
        if meta_root:
            plt.savefig(os.path.join(meta_root,'month_summary.png'),bbox_inches='tight')
        
        return agg_m
    
    @staticmethod
    def _get_country_df(country_name,df):
        fl = df['country'].apply(lambda x: country_name in x)
        df_c = df[fl]
        agg_df_c = df_c[['date','month']].groupby('month').agg('count')
        agg_df_c.columns =[country_name]
        return agg_df_c
    
    
    def export_country_stats(self,df,country_dict,out_dir):
        country_df_list = [self._get_country_df(c,df) for c in list(country_dict.keys())]
        country_agg_df = pd.concat(country_df_list,axis=1)
        
        print(country_agg_df.mean())
        
        if out_dir:
            export_file = os.path.join(out_dir,'meta_summary.xlsx')
            with pd.ExcelWriter(export_file) as writer:
                #agg_m.to_excel(writer,sheet_name='overall')
                country_agg_df.to_excel(writer,sheet_name='country_level')
        
        
#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', 
                        default='/data/News_data_raw/Production/data/input_processed_current_month/')
    parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', 
                        default='/data/News_data_raw/Production/data/meta/')
    parser.add_argument('-p', '--period', action='store', dest='period', default='crisis')
    parser.add_argument('-v', '--verbose', action='store', dest='verbose', default=True)
    args = parser.parse_args()
    
    ## initiate meta generator 
    mg = Meta_processor(MetaStreamer,crisis_points)
    ## create meta
    df_meta = mg.generate_meta(args.in_dir,args.out_dir)
    ## add a temp filter
    df_meta = df_meta[df_meta['month']>'2019-03']
    df_meta.to_pickle(os.path.join(args.out_dir, 'doc_details_{}.pkl'.format(args.period)))
    ## export summary statistics
    mg.create_summary(df_meta,meta_root=args.out_dir)
    mg.export_country_stats(df_meta,country_dict,args.out_dir)