# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:14:55 2023

@author: CHuang
"""
#%%
import os,sys
sys.path.insert(0, './libs')
from pygooglenews_ch import GoogleNews ## some customized fixes 
from utils import retry
#from bs4 import BeautifulSoup
import pandas as pd
import time,random
import json,os
from tqdm import tqdm

#%%
def chunk_list(input_list, n):
    """
    This function takes a list and a number 'n' as inputs.
    It returns a new list where the original list is divided into chunks, each with 'n' elements.
    """
    return [input_list[i:i + n] for i in range(0, len(input_list), n)]

def sliding_pairs(lst, n=2):
    """
    This function takes a list and a window size as inputs.
    It returns a new list where each element is a slice of the original list of size n.
    """
    step=n-1
    return [lst[i:i+n] for i in range(0, len(lst) - n + 1, step)]

def date_periods(start, end, pair=False, stride = 2):
    """
    This function takes start and end dates as inputs.
    It returns a list of periods with 1 month increments.
    """
    dates = pd.date_range(start=start, end=end, freq='MS')
    date_list = [date.strftime('%Y-%m-%d') for date in dates]
    
    if pair:
        date_list = sliding_pairs(date_list,n=stride)
    
    return date_list

def transfrom_keywords(k):
    if len(k.split())>1:
        k = '\"{}\"'.format(k)
    
    return k 

def run_one_test():
    periods = date_periods(start='2023-2-01', end='2023-05-01',pair = True,stride=3)
   
    #print(periods[:5])
    key_list = ['copper',
                 'lithium',
                 'nickel',
                 'cobalt',
                 '"rare earth elements"',
                 'zinc',
                 '"rare earths"',
                 'REE']
    res = run_one_query_with_keys(key_list,periods[0],site='bloomberg.com',verbose=True)
    return res

@retry(attempts=3, delay=3)
def run_one_query_with_keys(key_list,time_interval,site=None,
                            verbose=False,scraping_bee_key=None,
                            retry=True):
    query= ' OR '.join(key_list)
    if site:
        query += " site:{}".format(site)
    s,e = time_interval[0],time_interval[-1]
    if scraping_bee_key:
        res = gn.search(query,from_ = s, to_ = e,scraping_bee=scraping_bee_key)
    else:
        res = gn.search(query,from_ = s, to_ = e)
    
    if verbose:
        print(query)
        print('period: {} - {}'.format(s,e))
        print('returned results : {}'.format(len(res['entries'])))
        if len(res['entries'])>0:
            print(res['entries'][0]['title'])
    
    if retry:
        if len(res['entries'])==0:
            raise Exception('Warning: no article found for this query')
        
    return res

def run_by_keychunks(k_chunks,time_interval,site=None,verbose=False,scraping_bee_key=None):
    agg_results = []    
    for kc in anchor_keywords_chunks:
        try:
            res = run_one_query_with_keys(kc,time_interval,
                                        site=site,
                                        verbose=verbose,
                                        scraping_bee_key=scraping_bee_key)
            if len(res.get('entries'))>0:
                agg_results.extend(res['entries'])
        except:
            print('Warning: no article found for this try')
            
        if not scraping_bee_key:
            wait_time = random.randint(10, 50)
            time.sleep(wait_time)
            
    return agg_results

def read_newspapers(input_path):
    websites = pd.read_excel(input_path,sheet_name='newspaper')['newspaper_website'].values.tolist()
    key_list = pd.read_excel(key_words_p, sheet_name='search_key')['search_key'].values.tolist()
    key_list = [transfrom_keywords(k) for k in key_list]

    return websites, key_list

def remove_duplicates(dicts_list,check_key='link'):
    """
    check for duplicate values 
    """
    seen_links=set()
    unique_dicts=[]
    for d in dicts_list:
        link_value = d.get(check_key)
        if link_value not in seen_links:
            seen_links.add(link_value)
            unique_dicts.append(d)
    
    return unique_dicts

def run_one_website(site,periods,anchor_keywords_chunks,output_f):
    for time_interval in tqdm(periods):
        #print('working on {}'.format(time_interval))
        #out_name = '{}_{}_{}.json'.format(site.split('.')[0],time_interval[0],time_interval[-1])
        out_name = '{}_{}_{}.json'.format(site,time_interval[0],time_interval[-1])
        res_list = run_by_keychunks(anchor_keywords_chunks,time_interval,
                                    site=site,verbose=True,scraping_bee_key=None)
        res_list = remove_duplicates(res_list,check_key='link')
        with open(os.path.join(output_f,out_name), 'w',encoding='utf8') as f:
            json.dump(res_list, f, indent=4)


#%%
if __name__ == "__main__":
    
    TEST = False
    key_words_p = r'/data/chuang/news_scrape/data/inputs.xlsx'
    output_f = r'/data/chuang/news_scrape/data/raw'
    start = '2010-01-01'
    end = '2023-01-01'
    # news_papers= {'bloomberg.com':2, ## stride = 2, means interval is every month 
    #               'reuters.com':2,
    #               'nytimes.com':2,
    #               'wsj.com':2,
    #               'economist.com':12,
    #               'cnbc.com':2}
    #site = 'reuters.com'
    #site = 'thestkittsnevisobserver.com'
    stride = 3  # set it to 4, run every quarter
    periods = date_periods(start,end,pair=True,stride=stride)
    websites, key_list = read_newspapers(key_words_p)
    anchor_keywords_chunks = chunk_list(key_list,n=3)
    #%%
    ## initiate google news agent
    gn = GoogleNews(lang = 'en')  #, country = 'US'
    
    if TEST:
        run_one_test()
    else:
        for site in websites:
            print('working on site: {}'.format(site))
            run_one_website(site,periods,anchor_keywords_chunks,output_f)


        

    
    # #%%
    # with open('data.json', 'r',encoding='utf8') as f:
    #     data = json.load(f)

# %%
