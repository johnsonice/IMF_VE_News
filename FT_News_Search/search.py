#%%
import os, sys
sys.path.insert(0,'libs')
import glob 
from utils import get_all_files,read_json,construct_rex,dicts_to_jsonl,save2pickle,merge_dict_keys
import pathlib
import re
import pandas as pd
from datetime import datetime as dt
from joblib import Parallel, delayed
import joblib
from collections import Counter
#%%
def read_company_keys(search_key_path = 'search_keys/company_name.xlsx'):
    
    key_df = pd.read_excel(search_key_path)
    key_df['CompanyName'] = key_df['CompanyName'].str.strip()
    key_df = key_df.drop_duplicates(subset='CompanyName', keep="first")
    s_keys = key_df['CompanyName'].to_list()
    return s_keys

#%%
def find_exact_keywords(content,keywords,rex=None):
    if rex is None: 
        rex = construct_rex(keywords,plural=False,case=False)
    content = content.replace('\n', '').replace('\r', '')#.replace('.',' .')
    match = Counter([m.group() for m in rex.finditer(content)])             # get all instances of matched words 
                                                                            # and turned them into a counter object, to see frequencies
    total_count = sum(match.values())
    return match,total_count

def match_keywords_from_json(inp,rex,verbose=False):
    ## part input 
    idx,row = inp
    ## read file 
    article = read_json(row['file_path'])
    ## get all match content
    search_content = []
    content_keys = ['title','snippet','body']
    for ck in content_keys:
        cc = article.get(ck)
        if cc:
            search_content.append(cc)
    search_content = " ".join(search_content)
    ## use regex to fine keywords 
    match_dict,total_count = find_exact_keywords(search_content,None,rex)
    ## format return results 
    if total_count == 0:
        return None
    else:
        res = row.to_dict()
        match_dict = merge_dict_keys(match_dict)
        res.update(match_dict)
        return res

#%%
if __name__=="__main__":
    wd_path = '/data/chuang/Financial_Times/Working_directory'
    meta_path = os.path.join(wd_path,'ft_meta.pkl')
    search_key_path = 'search_keys/company_name.xlsx'
    search_out_josnl_path = os.path.join(wd_path,'search_results','search_raw.jsonl')
    search_out_pickle_path = os.path.join(wd_path,'search_results','search_raw.pkl')
    search_out_year_path = os.path.join(wd_path,'search_results','search_raw_{}')
    #%%
    meta_df = pd.read_pickle(meta_path)
    meta_df['year'] = meta_df['date'].apply(lambda i: i.year) 
    all_years = meta_df['year'].unique()
    #%%
    #search_keys = ['apple','bank','financial']
    search_keys = read_company_keys(search_key_path)
    search_rex = construct_rex(search_keys,plural=False,case=False)

    #%%
    number_of_cpu = joblib.cpu_count() - 4
    parallel_pool = Parallel(n_jobs=number_of_cpu,verbose=5)
    #rows = meta_df.head(10000).iterrows()
    #%%
    all_match_res = []
    for y in all_years:
        meta_chunk = meta_df[meta_df['year']==y]
        if len(meta_chunk) > 0:
            print('\n.... Processing Year : {} total # docs : {} ....\n'.format(y,len(meta_chunk)))
            #meta_chunk = meta_chunk.head(1000)
            rows = meta_chunk.iterrows()
            delayed_funcs = [delayed(match_keywords_from_json)(x,search_rex) for x in rows]
            match_res = parallel_pool(delayed_funcs)
            match_res = [m for m in match_res if m is not None]
            all_match_res.extend(match_res)
            #dicts_to_jsonl(match_res,search_out_year_path.format(str(y)+'.jsonl'))
            save2pickle(match_res,search_out_year_path.format(str(y)+'.pkl'))

    ## save temp results to pickle and jsonl 
    #dicts_to_jsonl(all_match_res,search_out_josnl_path)
    save2pickle(all_match_res,search_out_pickle_path)
    #%%
    del delayed_funcs   ## clear memory 
    del parallel_pool   ## clear memory 

    #%% if pandas can handle it, try turn it into dataframe as well 
    ## export results 
    res_df = pd.DataFrame(all_match_res)
    res_df.to_csv(os.path.join(wd_path,'ft_company_match.csv'))
    res_df.to_pickle(os.path.join(wd_path,'ft_company_match.pkl'))
    print('export to {}'.format(wd_path))
    # %%


    # #%% run single process example
    # match_res = []
    # rows = meta_df.head(10000).iterrows()
    # for r in rows:
    #     m_r = match_keywords_from_json(r,search_rex,verbose=False)
    #     match_res.append(m_r)
