## merge all news data 

#%%
import os, sys
sys.path.insert(0, './libs')
from utils import get_all_files,read_json
import pandas as pd
##newpaper docs
##https://newspaper.readthedocs.io/en/latest/user_guide/advanced.html#parameters-and-configurations
from tqdm import tqdm
from search_util import (
    construct_rex,
    process_keywords_with_logic,
    separate_overlapping,
    find_exact_keywords_with_overlaps
    )
import numpy as np

def merge_all_news(input_file_paths,output_path=None):
    """
    input_files assumes to be json with specific formate
    """
    keep_info=[]
    for f in tqdm(file_ps):
        news,time_start,time_end = f.split("_")
        time_end= time_end.split('.')[0]
        file_p = os.path.join(data_folder,f)
        data = read_json(file_p)
        for d in data:
            year,month,day = d.get('published_parsed')[:3]
            temp_res = (news,time_start,time_end,year,month,day,d.get('title'),d.get('link'))
            ext = d.get('extracted_content')
            if ext:
                news_body = ext.get('bodyXML')
            else:
                news_body=''
            temp_res = temp_res + (news_body,)
            keep_info.append(temp_res)
    
    df = pd.DataFrame(keep_info,columns=['newspaper_name','time_start','time_end','year',
                                         'month','day','title','link','body'])
    
    ## remove duplicates by link , keep the one with largest length 
    df['body_length'] = df['body'].str.len()
    df = df.sort_values(by='body_length',ascending=False)
    df = df.drop_duplicates(subset='link',keep='first')
    df = df.sort_values(by=['newspaper_name','year',
                                      'month','day'],ascending=True)

    if output_path:
        df.to_csv(output_path,index=False,encoding='utf8')
        print('export results to {}'.format(output_path))

    return df

def run_search(df,rex_groups,all_search_keywords,logical_keys,keywords_dict):
    """
    do keywords match in a dataframe 
    """

    orginal_columns = df.columns.to_list()
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.len() > 5]
    df['search_res'] = df['text'].str.lower().apply(find_exact_keywords_with_overlaps,
                                                    rex_groups=rex_groups,
                                                    return_count=False)   
    df = df.join(pd.json_normalize(df.pop('search_res'))).fillna(0)
    matched_cols = [k for k in df.columns if k in all_search_keywords] ## all are lower case, so we are fine here  
    df[matched_cols] = df[matched_cols].applymap(lambda x: 1 if x >= 1 else 0)
    
    for lk in logical_keys:
        ks = [k.strip() for k in lk.split(and_key)]
        if all([k in matched_cols for k in ks]):
            df[lk] = df[ks].sum(axis=1)
            df[lk] = np.where(df[lk] == len(ks), 1, 0)
        else:
            df[lk] = 0 
    ## drop original columns 
    for k in ks:
        if k in matched_cols:
            df.drop([k],axis=1)

    ## clean up and add dummy for keep 
    orginal_columns.remove('body')
    matched_cols = [k for k in df.columns if k in keywords_dict['search_key']] 
    df = df[orginal_columns+matched_cols]
    df['Keep_tag'] = df[matched_cols].sum(axis=1).clip(upper=1)

    return df
#%%
if __name__ == "__main__":
    #raw_data_f = '/data/chuang/news_scrape/data/raw_try1'
    key_words_p = r'/data/chuang/news_scrape/data/inputs.xlsx'
    data_folder = '/data/chuang/news_scrape/data/news_with_body'
    res_folder = '/data/chuang/news_scrape/data/news_search_res'
    news_output_p = os.path.join(res_folder,'news_merged.csv')
    search_res_output_p = os.path.join(res_folder,'search_results_raw.csv')
    news_agency = None # none means get them all 'thestkittsnevisobserver'
    ## get remaining files to download 
    file_ps = get_all_files(data_folder,end_with='.json',start_with=news_agency,return_name=True) ## cnbc bloomberg reuters
    
    MERGE_NEWS=True

    #%%
    if MERGE_NEWS:
        news_df = merge_all_news(file_ps,output_path=news_output_p)
    else:
        news_df = pd.read_csv(news_output_p)
    
    #%%

    ## read search keys and construct match pattern 
    and_key = '\+'
    keywords_dict , all_search_keywords, logical_keys = process_keywords_with_logic(key_words_p,
                                                                                    search_sheet_name='post_search_key')
    g1,g2 = separate_overlapping(all_search_keywords)
    search_rex_1 = construct_rex(g1,casing=False,plural=False)  ## here we are using case insensitive
    search_rex_2 = construct_rex(g2,casing=False,plural=False)
    #%%
    ## run one test 
    res = find_exact_keywords_with_overlaps('this is about climate change review Climate policy',
                              rex_groups=[search_rex_1,search_rex_2],
                              return_count=False)
    print(res)
    #%%
    news_df['text'] = news_df['title'] + " ; "+news_df['body'] 
    res_df = run_search(news_df,[search_rex_1,search_rex_2],all_search_keywords,logical_keys,keywords_dict)
    res_df.to_csv(search_res_output_p,index=False,encoding='utf8')
    print('export search results to {}'.format(search_res_output_p))
# %%
