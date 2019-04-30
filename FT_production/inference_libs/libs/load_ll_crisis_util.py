# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:51:36 2019

@author: CHuang
"""

import pandas as pd
#%%

def transform_dates(date_str,default_m='01'):
    ds = str(date_str).strip()
    dl = ds.split('&')
    new_dl = []
    for d in dl:
        if len(d)<4 or len(d)>7:
            raise Exception('date format is wrong: {}'.format(d))
        elif '-' in d:
            new_dl.append(d.strip())
        else:
            new_dl.append(d+'-'+default_m)
    
    return new_dl

def get_ll_crisis_points(file_path,sheet_name,country_filter=None):
    df = pd.read_excel(file_path,sheets=sheet_name)
    df['starts'] = df['start'].apply(transform_dates,args=('01',))
    df['peaks'] = df['end'].apply(transform_dates,args=('12',))
    df['imf_country'] = df['imf_country'].apply(lambda x: x.lower())
    ll_crisis = zip(df['imf_country'],df['starts'],df['peaks'])
    ll_crisis_dict = {c:{'starts':s,'peaks':p} for c,s,p in ll_crisis}
    if country_filter:
        ll_crisis_dict = {k:v for (k,v) in ll_crisis_dict.items() if k in country_filter}
    return ll_crisis_dict

#%%
if __name__ =="__main__":
    fp = 'll_crisis_dates.xlsx'
    sheet_name = 'import'
    res = get_ll_crisis_points(fp,sheet_name,['sweden','thailand','turkey'])
    
