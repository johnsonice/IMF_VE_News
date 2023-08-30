# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:30:42 2023

@author: CHuang
"""
#%%
import os,sys
sys.path.insert(0, './libs')
import newspaper
from newspaper import Config, Article
from utils import get_all_files,read_json,list_difference_left,retry
##newpaper docs
##https://newspaper.readthedocs.io/en/latest/user_guide/advanced.html#parameters-and-configurations
import base64,re
from tqdm import tqdm
import json
import time 
import os, sys , ssl

# ssl._create_default_https_context = ssl._create_unverified_context
# os.environ['PYTHONHTTPSVERIFY'] = "0"

# if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
#     ssl._create_default_https_context = ssl._create_unverified_context
    
#%%
def clean_decoded_url(dec_url):
    clean_url = 'http'+dec_url.split('http')[1]
    clean_url = clean_url[:-2]
    return clean_url

def decode_google_news_url(encoded_url,clean=True):
    # Split the URL by '/' and get the portion that appears to be base64-encoded
    possible_encoded_url = encoded_url.split('/')[-1]
    # The URL might have query parameters after '?', we should remove it.
    possible_encoded_url = possible_encoded_url.split('?')[0]
    possible_encoded_url += "=" * ((4 - len(possible_encoded_url) % 4) % 4) #ugh

    # Decode the base64-encoded string
    decoded_bytes = base64.urlsafe_b64decode(possible_encoded_url)

    # Convert the bytes back to a string
    decoded_url = decoded_bytes.decode('utf-8',errors='ignore')
    
    if clean:
        decoded_url = clean_decoded_url(decoded_url)
        
    return decoded_url

def _article2dict(article,style='ft'):
    """
    convert article object to dictionary or to financial times formate 
    """
    if style =='ft':
        res_dict ={
                    "webUrl": article.url,
                    "has_body": article.is_valid_body(),
                    "title": str(article.title),
                    "bodyXML": str(article.text),
                    "authors": article.authors,
                    "publishedDate": str(article.publish_date),
                    "keywords": article.keywords,
                    "standfirst": str(article.summary)
                }
    else:
        res_dict ={
                    "link": article.url,
                    "title": str(article.title),
                    "text": str(article.text),
                    "authors": article.authors,
                    "published_date": str(article.publish_date),
                    "top_image": str(article.top_image),
                    "videos": article.movies,
                    "keywords": article.keywords,
                    "summary": str(article.summary),
                    "has_body": article.is_valid_body()
                    }
        
    return res_dict

@retry(attempts=5, delay=2)
def get_news_article(url,to_dict=True,ScrapingBee_client=None,dict_stype='ft'):
    if not ScrapingBee_client:
        article = newspaper.Article(url=url, language='en',request_timeout=15)
        article.download()
        article.parse()
        
        if to_dict:
            article =_article2dict(article,dict_stype)
    else:

        response = ScrapingBee_client.get(
                                            url, 
                                            params={'custom_google': 'False'},
                                        )
        
        article = Article('',language='en')
        if response.status_code == 200:
            article.download(input_html = response.text)
            article.parse()
            article.url = url
            if to_dict:
                article = _article2dict(article,dict_stype)       
        else:
            raise Exception('Warning: download failded:{}'.format(url))
            
    return article


#%%
if __name__ == "__main__":
    raw_data_f = '/data/chuang/news_scrape/data/raw'
    res_data_f = '/data/chuang/news_scrape/data/news_with_body'
    news_agency = 'thestkittsnevisobserver'
    ## get remaining files to download 
    file_ps = get_all_files(raw_data_f,end_with='.json',start_with=news_agency,return_name=True) ## cnbc bloomberg reuters
    downlaoded_ps = get_all_files(res_data_f,end_with='.json',start_with=news_agency,return_name=True)
    file_ps = list_difference_left(file_ps,downlaoded_ps)
    file_ps = [os.path.join(raw_data_f,f_n) for f_n in file_ps]
    #%%
    for file_p in file_ps:
        j_name = os.path.basename(file_p)
        print(j_name)
        out_p = os.path.join(res_data_f,j_name)
        data = read_json(file_p)
        for i in tqdm(data):
            link = i['link']
            real_link = decode_google_news_url(link,clean=True)
            try:
                doc = get_news_article(real_link,to_dict=True,dict_stype='ft')
            except Exception as e:
                doc = {}
                print(e)
            # print(doc['title'])
            # print(doc['text'])
            i['extracted_content'] = doc
            time.sleep(1)
        with open(out_p, 'w',encoding='utf8') as f:
            json.dump(data, f, indent=4)
    #%%
    
    # ### test see if link is extracted correctly 
    # file_ps = get_all_files(raw_data_f,end_with='.json',start_with='wsj')
    # data = read_json(file_ps[0])
    # for i in data[:50]:
    #     link = i['link']
    #     real_link = decode_google_news_url(link,clean=True)
    #     print(real_link)
    
    #%%
    # link = 'https://www.cnn.com/2023/01/24/us/mass-shootings-fast-facts'
    # doc = Article(url=link, language='en',request_timeout=5,
    #               memoize_articles=False,
    #               proxy='')
    # doc.download()
    # doc.parse()

    # #%%
    # a = get_news_article(link,to_dict=True,dict_stype='ft')
    #%%
    
    # from scrapingbee import ScrapingBeeClient
    # client = ScrapingBeeClient(api_key='xxxxxxxxxxxxxxxxxxx')
    
    #%%
    
    
