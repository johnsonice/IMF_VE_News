# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:21:13 2022

@author: CHuang
"""

import re
from collections import Counter
import pandas as pd 

def get_keywords_groups(key_path,clean=False,clean_keys=None,sheet_name=None,lower=True):
    if sheet_name:
        key_df = pd.read_excel(key_path,sheet_name=sheet_name)
    else:
        key_df = pd.read_excel(key_path)
        
    key_group_dict = key_df.to_dict('list')
    for k in key_group_dict.keys():
        if lower:
            key_group_dict[k] = [i.strip('\xa0').lower() for i in key_group_dict[k] if not pd.isna(i)]  
        else:
            key_group_dict[k] = [i.strip('\xa0') for i in key_group_dict[k] if not pd.isna(i)]    
        
        if clean:
            ## if clean keys function was passed, process keywords 
            key_group_dict[k] = clean_keys(key_group_dict[k])
            
    return key_group_dict


def construct_rex(keywords,plural=True,casing=False):
    """
    construct regex for multiple match 
    """
    if plural:
        r_keywords = [r'\b' + re.escape(k) + r'(s|es)?\b'for k in keywords]    # tronsform keyWords list to a patten list, find both s and es 
    else:
        r_keywords = [r'\b' + re.escape(k) + r'\b'for k in keywords]
    
    if casing:
        rex = re.compile('|'.join(r_keywords)) 
    else:
        rex = re.compile('|'.join(r_keywords),flags=re.I)                       # use or to join all of them, ignore casing
        #match = [(m.start(),m.group()) for m in rex.finditer(content)]         # get the position and the word
    return rex
    

def construct_rex_group(key_group_dict):
    """
    construct a group of regular expression patterns 
    """
    reg_dict = {}
    for k in key_group_dict.keys():
        reg_dict[k] = construct_rex(key_group_dict[k])
        
    return reg_dict

def find_exact_keywords(content,keywords=None,content_clean=True,rex=None,return_count=True):
    if rex is None: 
        rex = construct_rex(keywords)
    
    if content_clean:
        content = content.replace('\n', '').replace('\r', '')#.replace('.',' .')
    
    match = Counter([m.group() for m in rex.finditer(content)])             # get all instances of matched words 
                                                                            # and turned them into a counter object, to see frequencies
    total_count = sum(match.values())
    
    if return_count:
        return match,total_count
    else:
        return match

def merge_dict_keys(org_dict:dict):
    """
    merge dict keys with different cases and lower everything
    """
    merged_dict = {}
    for k,v in org_dict.items():
        if isinstance(k,str):
            if merged_dict.get(k.lower()):
                merged_dict[k.lower()] += org_dict[k]
            else:
                merged_dict[k.lower()] = org_dict[k]
        else:
            merged_dict[k] = org_dict[k]
    return merged_dict

def separate_overlapping(strings):
    # Sort strings by length in descending order
    strings.sort(key=len, reverse=True)
    
    list1 = []
    list2 = []
    
    for string in strings:
        # Check if string is a substring of any string in list1
        if any(string in s for s in list1):
            list2.append(string)
        else:
            list1.append(string)
    
    return list1, list2

def merge_dicts(dict_list):
    merged_dict = {}
    for dictionary in dict_list:
        merged_dict.update(dictionary)
    return merged_dict

def find_exact_keywords_with_overlaps(content,rex_groups,
                                      content_clean=True,
                                      return_count=True,
                                      all_lower=True):
    """
    with there are overlapping issues in the keywords list provided, 
    we need to seperate those overlapping keywrods into a seperate lsit 
    and do them iteratively to make sure all matches are returned
    this function is to faciliate that operation 
    """
    res_dict_list = []
    res_total_count = 0 
    for rex in rex_groups:
        match,total_count = find_exact_keywords(content,rex=rex,
                                                content_clean=content_clean,return_count=True)
        res_dict_list.append(match)
        res_total_count+=total_count
    
    res_dict = merge_dicts(res_dict_list)
    if all_lower:
        res_dict = merge_dict_keys(res_dict)
    if return_count:
        return res_dict,res_total_count
    else:
        return res_dict

def process_keywords_with_logic(seasrch_key_files,return_merged_keys=True,
                                return_logic=True,and_key='\+',key_lower=True,
                                search_sheet_name=None):
    """
    Parameters
    ----------
    seasrch_key_files : file path; string
    Returns
    -------
    kg : Dict
        a dict of keywords groups 
    all_keys : List
        all search key merged together
    """
    kg = get_keywords_groups(seasrch_key_files,lower=key_lower,sheet_name=search_sheet_name)
    keys_nested = list(kg.values())
    
    ## remove duplicates 
    all_keys = list(set([item for sublist in keys_nested for item in sublist]))
    
    ## break keys with logics 
    filtered_list = [item for item in all_keys if and_key not in item]
    logical_keys = [item for item in all_keys if item not in filtered_list]
    
    ## process logical keys and merge them back 
    if len(logical_keys)>0:
        logical_keys_split = [item.split(and_key) for item in logical_keys]
        logical_keys_split = list(set([item.strip() for sublist in logical_keys_split for item in sublist]))
        
        filtered_list.extend(logical_keys_split) ## merge logical terms back togeher 
        filtered_list = list(set(filtered_list)) ## remove duplicates again 
        
    return kg,filtered_list, logical_keys