#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:00:33 2018

@author: chuang
"""
## add country metadata 
## data exploration, descripitive analysis 

import sys,os
sys.path.insert(0,'./libs')
import config
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from crisis_points import country_dict
from nltk.tokenize import word_tokenize
from stream import MetaStreamer_fast as MetaStreamer
#import time 
from mp_utils import Mp
import re
import logging
#plt.rcParams['figure.figsize']=(10,5)

global min_this
global max_other
global other_type
global top_n
global country_dict

# Experimental country classification variables
min_this = 1
max_other = None
other_type = "sum"
top_n = None
country_dict = config.country_dict
f_handler = logging.FileHandler('err_log_7_1908.log')
f_handler.setLevel(logging.WARNING)

#%%
## save quarterly figure
def create_summary(agg_q,meta_root, pass_name):
    x_ticker = agg_q.index[0::4]
    ax = agg_q.plot(figsize=(16,6),title='News Articles Frequency',legend=False)
    plt.ylabel('Number of news articles')       ## you can also use plt functions 
    plt.xlabel('Time-Q') 
    ax.set_xticks(x_ticker)                     ## set ticker
    ax.set_xticklabels(x_ticker,rotation=90)    ## set ticker labels
    ax.get_xaxis().set_visible(True)            ## some futrther control over axis
    plt.savefig(os.path.join(meta_root,'quarter_summary_{}.png'.format(pass_name)),bbox_inches='tight')

#%%
#def get_country_name(tokens,country_dict):
#    for c,v in country_dict.items():
#        rc = c if tokens and any([tok.lower() in tokens for tok in v]) else None
#        if rc is not None:
#            yield rc 

#def get_countries(article,country_dict=country_dict):
#    snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
#    title = word_tokenize(article['title'].lower()) if article['title'] else None
#
#    if snip and title:
#        title.extend(snip)
#        cl = list(get_country_name(title,country_dict))
#    elif title:
#        cl = list(get_country_name(title,country_dict))
#    elif snip:
#        cl = list(get_country_name(snip,country_dict))
#    else:
#        cl = list()
#        
#    return article['an'],cl    

def construct_rex(keywords,case=False):
    r_keywords = [r'\b' + re.escape(k)+ r'(s|es|\'s)?\b' for k in keywords]
    if case:
        rex = re.compile('|'.join(r_keywords)) #--- use case sentitive for matching for cases lik US
    else:
        rex = re.compile('|'.join(r_keywords),flags=re.I) ## ignore casing
    return rex


def get_country_name(text,country_dict,rex=None):
    for c,v in country_dict.items():
        if c in ['united-states']:
            rex = construct_rex(v,case=True)
        else:
            rex = construct_rex(v)
        rc = rex.findall(text)
        if len(rc)>0:
            yield c


def get_countries(article,country_dict=country_dict):
    #snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    #title = word_tokenize(article['title'].lower()) if article['title'] else None
    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    if snip and title:
        #title.extend(snip)
        title = "{} {}".format(title,snip)
        cl = list(get_country_name(title,country_dict))
    elif title:
        cl = list(get_country_name(title,country_dict))
    elif snip:
        cl = list(get_country_name(snip,country_dict))
    else:
        cl = list()
        
    return article['an'],cl


def get_country_name_count(text, country_dict=country_dict, min_count=min_this, rex=None):
    for c, v in country_dict.items():
        if c in ['united-states']:
            rex = construct_rex(v, case=True)
        else:
            rex = construct_rex(v)
        rc = rex.findall(text)
        l_rc = len(rc)
        if l_rc > 0 and l_rc >= min_count:
            yield c


def get_country_name_count_2(text):
    verbose = False  #TEMP
    name_list = []
    num_list = []

    if verbose:
        print("\tMin_this :", min_this)
        print("\tMax_other:", max_other)
        print("\tOther_type:", other_type)
        print("\ttop_n:", top_n)

    for c, v in country_dict.items():
        if c in ['united-states']:
            rex = construct_rex(v, case=True)
        else:
            rex = construct_rex(v)
        rc = rex.findall(text)
        l_rc = len(rc)
        if l_rc > 0:
            name_list.append(c)
            num_list.append(l_rc)

    if verbose:
        print("Raw names: {}".format(name_list))
        print("Raw nums: {}".format(num_list))

    if max_other is None:
        pruned_name_list = name_list
        pruned_num_list = num_list
    # Eliminate a country if other countries dominate
    else:
        pruned_name_list = []
        pruned_num_list = []

        # Based on sum of other country instances
        if other_type == "sum":
            country_mentions = sum(num_list)
            for i in range(len(name_list)):
                if country_mentions - num_list[i] <= max_other:
                    pruned_name_list.append(name_list[i])
                    pruned_num_list.append(num_list[i])

        # Based on a threshold amount for any other country
        elif other_type == "at_least":
            garbage_index = []
            for i in range(len(name_list)):
                exl_list = num_list[:i]  # Save number of instances of other countries
                if i < len(num_list)-1:
                    exl_list = exl_list + num_list[i+1:]

                for o_num in exl_list:
                    if o_num > max_other:
                        garbage_index.append(i)  # Track which countries to remove
                        continue

            # Save the pruned list
            for add_ind in range(len(name_list)):
                if add_ind not in garbage_index:
                    pruned_name_list.append(name_list[add_ind])
                    pruned_num_list.append(num_list[add_ind])

    if verbose:
        print('Pruned 1 names: {}'.format(pruned_name_list))
        print('Pruned 1 nums: {}'.format(pruned_num_list))

    # Eliminate countries that show up less than the threshold amount (default once)
    double_pruned_names = []
    double_pruned_nums = []
    for i in range(len(pruned_name_list)):
        if pruned_num_list[i] >= min_this:
            double_pruned_names.append(pruned_name_list[i])
            double_pruned_nums.append(pruned_num_list[i])

    if verbose:
        print("Pruned 2 names: {}".format(double_pruned_names))
        print("Pruned 2 nums: {}".format(double_pruned_nums))

    # If want to keep only the top n most encountered countries
    triple_pruned = []
    if top_n is not None:
        zipped = list(zip(double_pruned_names, double_pruned_nums))
        zipped.sort(key=lambda x: x[1])

        i = 0
        while i < len(double_pruned_names):
            if len(triple_pruned) >= top_n:
                break
            else:
                top_append = []
                going_num = double_pruned_nums[i]
                for j in range(len(double_pruned_nums[i:])):  # Allow more countries if tied for place
                    if double_pruned_nums[j] != going_num:
                        break
                    else:
                        top_append.append(double_pruned_names[j])
            triple_pruned.extend(top_append)
            if len(top_append) > 0:
                i = i + len(top_append)
            else:
                i = i + 1
    else:
        triple_pruned = double_pruned_names

    if verbose:
        print("Pruned 3 names: {}".format(triple_pruned))

    return triple_pruned


def get_countries_by_count(article):
    '''
    Identifies list of countries based on number of instances of this country except for if other countries
        show up >y times.

    :param article: the article to classify
    :param country_dicts: countries to look through
    :param min_this: minimum observations of a country (default 1)
    :param max_other: maximum other country mentions allowed if None, ignore (default None)
    :param other_type: Only if max_other is not None,
    :return:
    '''
    # snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    # title = word_tokenize(article['title'].lower()) if article['title'] else None
    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    if snip and title:
        # title.extend(snip)
        title = "{} {}".format(title, snip)
        cl = list(get_country_name_count(title))
    elif title:
        cl = list(get_country_name_count(title))
    elif snip:
        cl = list(get_country_name_count(snip))
    else:
        cl = list()

    return article['an'], cl


def get_countries_by_count_2(article):
    '''
    Identifies list of countries based on number of instances of this country except for if other countries
        show up >y times.

    :param article: the article to classify
    :param country_dicts: countries to look through
    :param min_this: minimum observations of a country (default 1)
    :param max_other: maximum other country mentions allowed if None, ignore (default None)
    :param other_type: Only if max_other is not None,
    :return:
    '''
    # snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    # title = word_tokenize(article['title'].lower()) if article['title'] else None
    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    if snip and title:
        # title.extend(snip)
        title = "{} {}".format(title, snip)
        cl = list(get_country_name_count(title))
    elif title:
        cl = list(get_country_name_count(title))
    elif snip:
        cl = list(get_country_name_count(snip))
    else:
        cl = list()

    return article['an'], cl


#%%
if __name__ == '__main__':
    meta_root = config.DOC_META
    meta_aug = config.AUG_DOC_META
    meta_pkl = config.DOC_META_FILE
    json_data_path = config.JSON_LEMMA
    
    df = pd.read_pickle(meta_pkl)

    country_dict = {
        'argentina': ['argentina'],
        'bolivia': ['bolivia'],
        'brazil': ['brazil'],
        'chile': ['chile'],
        'colombia': ['colombia']
    }

    class_type_setups = [
        ['Min1', 1, None, None, None],  # Country mentioned >= 1 times
        ['Min3', 3, None, None, None],  # Country mentioned >=3 times
        ['Min3_Max0', 3, 0, "sum", None],  # Country mentioned >=3 times, 0 other countries mentioned
        ['Min1_Max2_sum', 1, 2, "sum", None],  # Country mentioned >=1 time, <=2 mentions of other countries
        ['Min1_Top1', 1, None, None, 1],  # Country mentioned >=1 times, choose only the top 1 country
        ['Min3_Top1', 3, None, None, 1],  # Country mentioned >=3 times, choose only top 1 country
        ['Min1_Top3', 1, None, None, 3]  # Country mentioned >=1 times, choose only the top 3 countries
    ]

    df['data_path'] = json_data_path+'/'+df.index + '.json'
    print('see one example : \n',df['data_path'].iloc[0])
    pre_chunked = True  # The memory will explode otherwise
    for setup in class_type_setups:
        class_type = setup[0]
        # Configure run variables

        min_this = setup[1]
        max_other = setup[2]
        other_type = setup[3]
        top_n = setup[4]

        # Go through the files in chunks
        if pre_chunked:

            data_list = df['data_path'].tolist()
            pre_chunk_size = 50000
            chunky_index = 0
            data_length = len(data_list)
            index = []
            country_list = []
            while chunky_index < data_length:
                if chunky_index%100000 == 0:
                    print("Passed ", chunky_index, " files")
                chunk_end = min(chunky_index+pre_chunk_size, data_length)
                streamer = MetaStreamer(data_list[chunky_index:chunk_end])
                news = streamer.multi_process_files(workers=10, chunk_size=5000)
                mp = Mp(news, get_countries_by_count_2)
                country_meta = mp.multi_process_files(workers=10, chunk_size=5000)
                index = index + [i[0] for i in country_meta]
                country_list = country_list + [i[1] for i in country_meta]
                del country_meta  ## clear memory
                chunky_index = chunk_end

        # Eat all files at once
        else:
            streamer = MetaStreamer(df['data_path'].tolist())
            news = streamer.multi_process_files(workers=15, chunk_size=5000)
            # %%
            # country_meta = [(a['an'],get_countries(a,country_dict)) for a in news]
            mp = Mp(news, get_countries_by_count_2)
            country_meta = mp.multi_process_files(workers=15, chunk_size=5000)
            # %%
            index =[i[0] for i in country_meta]
            country_list =[i[1] for i in country_meta]
            del country_meta  ## clear memory

        ds = pd.Series(country_list,name='country',index=index)
        new_df = df.join(ds) ## merge country meta
        del ds  # Free space
        new_df['country_n'] = new_df['country'].map(lambda x: len(x))
        new_df.to_pickle(os.path.join(meta_aug, 'doc_details_{}_aug_{}.pkl'.format('crisis',class_type)))
        print('augumented document meta data saved at {}'.format(meta_aug))
    
        #%%
        # create aggregates for ploting
        agg_q = new_df[['date','quarter']].groupby('quarter').agg('count')
        del new_df  # Free space

        #agg_m = df[['date','month']].groupby('month').agg('count')
        create_summary(agg_q,meta_root,class_type)




#%%
