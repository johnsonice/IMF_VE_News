
#%%
import os, sys
#sys.path.insert(0,'libs')
import glob 
from libs.utils import get_all_files,read_json,construct_rex
import pathlib
from libs.country_dict_full import get_dict  
import re
import pandas as pd
from datetime import datetime as dt
# from joblib import Parallel, delayed
# import joblib
#%%
def construct_country_group_rex(country_dict):
    """
    construct a dictionary of regex patterns 
    """
    country_rex_dict = {}
    for c,v in country_dict.items():
        if c in ['united-states','united states of america']:
            rex = construct_rex(v,case=True)
        else:
            rex = construct_rex(v)
        
        country_rex_dict[c] = rex
    
    return country_rex_dict

def get_country_name(text,country_rex_dict):
    """
    use regex patterns to match 
    """
    for c,rex in country_rex_dict.items():
        rc = rex.findall(text)
        if len(rc)>0:
            yield c

def tag_country(article,country_rex_dict):
    """
    article is a dict object with title, snip, body and metadata 
    """
    ## look into 3 parts 
    stand = article.get('body').lower()[:200] if article.get('body') else ''
    title = article.get('title').lower() if article.get('title') else ''
    snip= article.get('snippet').lower() if article.get('snippet') else ''
    search_content = '. '.join([title,snip,stand])

    ## use regex to match countries 
    country_list = list(get_country_name(search_content,country_rex_dict))

    return country_list

def get_timestemp(article,file_path=None,date_format='ft',):
    """
    convert string to date object 
    """
    res = {}

    try:
        if date_format.strip().lower() == 'ft':
            date = pd.to_datetime(dt.strptime(article['publication_date'],'%Y-%m-%d'))
        else:
            date = pd.to_datetime(dt.fromtimestamp(article['publication_date'] / 1e3))
        res['date'] = date
        res['week'] = date.to_period('W')
        res['month'] = date.to_period('M')
        res['quarter'] = date.to_period('Q')
        return res

    except Exception as e:
        print(article['an'] + ': ' + str(e))
        return None

def get_all_meta_info(country_rex_dict,file_path=None,article=None,):
    meta_dict = {'file_path':file_path}
    
    if file_path:
        article = read_json(file_path)
    elif article:
        pass
    else:
        raise Exception('Need an input, either a json file path or a dict')

    if article:
        country_tags = list(tag_country(article,country_rex_dict))
        date_dict = get_timestemp(article)
        meta_dict.update(date_dict)
        meta_dict['country_name'] = country_tags

    return meta_dict

#%%

# if __name__=="__main__":
#     data_path = '/data/chuang/Financial_Times/Financial_Times_processed'
#     wd_path = '/data/chuang/Financial_Times/Working_directory'
#     country_dict = get_dict()
#     country_rex_dict = construct_country_group_rex(country_dict)

#     Flag_test = False
#     #%%
#     files = get_all_files(data_path,end_with='.json')
#     if Flag_test:
#         files = files[:5000]
#     print('Total length of documents : {}'.format(len(files)))
#     # %%
#     number_of_cpu = joblib.cpu_count() - 4
#     parallel_pool = Parallel(n_jobs=number_of_cpu,verbose=5)
#     delayed_funcs = [delayed(get_all_meta_info)(country_rex_dict,file_path=x) for x in files]
#     all_res = parallel_pool(delayed_funcs)
#     #%%
#     ## export 
#     df = pd.DataFrame(all_res)
#     df.to_csv(os.path.join(wd_path,'ft_meta.csv'))
#     df.to_pickle(os.path.join(wd_path,'ft_meta.pkl'))
#     print('export to {}'.format(wd_path))
    #%%

    #%%



    # %%
    # 
    # for f in files[:1000]:
    #     print(get_all_meta_info(country_rex_dict,file_path=f))
        # test = read_json(f)
        # if test:
        #     print(list(tag_country(test,country_rex_dict)),get_timestemp(test))