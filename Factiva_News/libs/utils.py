import os
import pathlib
import ujson as json
import re
import gzip
import json
import pickle
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
# from dotenv import load_dotenv
# env_path = '../../.env'
# load_dotenv(dotenv_path=env_path)

#%%

def download_hf_model(REPO_ID, save_location, hf_token=None):
    # REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    # save_location = '/root/data/hf_cache/llama-3-8B-Instruct'
    if hf_token is None:
        hf_token = os.getenv('huggingface_token')
        if hf_token is None:
            hf_token = input("huggingface token:")
            
    snapshot_download(repo_id=REPO_ID,
                    local_dir=save_location, 
                    token=hf_token)
    
    return save_location
    ## you can also use hf cli 
    ## huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B

def read_json(file_path, sample_size=None):
    """Load articles from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:sample_size] if sample_size else data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return [] 

def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

'''
    For the given path, get the List of all files in the directory tree 
'''
def get_all_files(dirName,end_with=None): # end_with=".json"
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_all_files(fullPath)
        else:
            allFiles.append(fullPath)
    
    if end_with:
        end_with = end_with.lower()
        allFiles = [f for f in allFiles if pathlib.Path(f).suffix.lower() == end_with ] 

    return allFiles   

def construct_rex(keywords,plural=True,case=False):
    if plural:
        r_keywords = [r'\b' + re.escape(k)+ r'(s|es|\'s)?\b' for k in keywords]
    else:
        r_keywords = [r'\b' + re.escape(k) for k in keywords]

    if case:
        rex = re.compile('|'.join(r_keywords)) #--- use case sentitive for matching for cases lik US
    else:  
        rex = re.compile('|'.join(r_keywords),flags=re.I) ## ignore casing 
    return rex
#%%

def dicts_to_jsonl(data_list: list, filename: str, compress: bool = False) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    sgz = '.gz'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data
    
    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w',encoding='utf-8') as compressed:
            for ddict in data_list:
                jout = json.dumps(ddict,ensure_ascii=False) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w',encoding='utf-8') as out:
            for ddict in data_list:
                jout = json.dumps(ddict, ensure_ascii=False) + '\n'
                out.write(jout)

def save2pickle(content,file_name:str):
    with open(file_name, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name:str):
    with open(file_name, 'rb') as handle:
        content = pickle.load(handle)

    return content 
#%%
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

