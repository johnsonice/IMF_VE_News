#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 07:59:21 2020

@author: chuang
"""

## general util functions
def write_to_txt(file_path,msg,over_write=True):
    
    if over_write:
        f = open(file_path,"w")
        f.write("{}".format(msg))
    else:
        f = open(file_path,"a")
        f.write("\n{}".format(msg))
    f.close()
    return None

def map_df_value(c_match_val,c_match,c_map,df,topn_index=0):
    """
    c_match_val : value yyou want to match 
    c_match: applied to which column
    c_map: from whcih column you want to get mapped value
    df: map as a dataframe
    """
    return df[df[c_match]==c_match_val][c_map].values[topn_index]