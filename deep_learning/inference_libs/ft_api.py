


import argparse
import requests
import os, sys, csv, re
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))
from timeit import default_timer as timer
import datetime as dt
import json
import time
import logging
import infer_config as config
#%%
# Set logging file name and logging parameters
logname = os.path.join(config.HISTORICAL_INPUT,"FT_Log_{}.log".format(str(dt.datetime.today().strftime("%m.%d.%Y"))))
logging.basicConfig(filename=logname,
                    format='%(asctime)s : %(levelname)s - %(message)s',
                    filemode='a',
                    level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
#%%
#########################################################################################
def get_params(filename, history,since=None):
    """
    Read file with API key [credentials_ft.txt] and the last saved page link [LAST_LINK.txt]* if available.
    If last saved page link is unavailable, use default url parameters with start date the first day of the current month.
    Return dictionary with query parameters.
    """
    with open(filename, 'r') as f:
        key = f.readline()
    api_key = key.strip()
    if history:
        with open(os.path.join(config.HISTORICAL_INPUT,'LAST_LINK.txt'),'r') as l:
            url1 = l.readline()
        main_url = url1.strip()
        timestamp = ""
        url_query = "{}&apiKey={}".format(main_url, api_key) 
    else:
        main_url = "https://api.ft.com/content/notifications?"
        if since:
            timestamp= since
        else:
            timestamp = get_current_month_begaining()
            logger.info('Since date not passed in, defaults to begaining of current month: {}'.format(since))
            #timestamp = '2019-01-01'
        timestamp = "{}T02:00:00.000Z".format(timestamp)
        url_query = "{}since={}&apiKey={}".format(main_url,timestamp, api_key)
    query_params = {
        'proxies':{'http': "socks5://intsquid:8000",'https': "socks5://intsquid:8000"},
        'base_url':main_url,
        'api_key': api_key,
        'start_date': timestamp,
        'url': url_query
        }
     
    return query_params

def generate_file(filename,data,data_dir):
    """Save content of the article and its metadata in json format to the homedir specified in argparse"""
    with open(os.path.join(data_dir,filename), 'w+') as outfile:
        outfile.write(json.dumps(data))#,indent=4, sort_keys=True))

def repeat_request(link,proxies,n=10):
    for i in range(n):
        try:
            r = requests.get(link, proxies=proxies)
            return r
        except:
            print('failed attempt {}'.format(i))
            time.sleep(5)
    
    print(link,proxies)
    raise Exception('Connection failed')
    
    #return r

def get_article_content(link,api_key,proxies,data_dir):
    """
    Get content of specified article. In case the content is unavailable [ERROR 403], log the error and article link.
    Continue iteration over article links on that page. There should be 200 links per page, unless the page is the latest one
    Returns name of the saved file or None
    """
    link = link+"?apiKey={}".format(api_key)
    logger.info("Article link: {}".format(link))

    #r = requests.get(link, proxies=proxies)
    r = repeat_request(link, proxies=proxies,n=5)
        
    if r.status_code == 403:
        logger.warning('Error with status code {}. Article is not available at the moment.'.format(r.status_code))
        return None
    elif r.status_code != 200:
        logger.warning('[?] Error: [HTTP {}]: Content: {}'.format(r.status_code, r.content))
        return None
    else:
        try:
            data = r.json()
            name_start = data['id'].split('thing/')[1]
            name_end = data['firstPublishedDate'].split('T')[0]
            file_name = name_start+"_"+name_end+".json"
            logger.info("Generated: {} ".format(file_name))
            generate_file(file_name,data,data_dir)
            return file_name
        except Exception as e:
            logger.info("error parsing article: {} ".format(e))
            return None

def iterate_through_pages(query_params,data_dir):
    """
    Retrieve proxies, api key, first url to query and get each article content.
    There has to be 200 articles per page unless it is a last page
    Iterate over pages using "next page" link provided on each page
    Once the "next page" link is unavaialble (same as the current link),
    return the list of values for metadata and all "next page" link
    """
    proxies = query_params.get('proxies', None)
    api_key= query_params.get('api_key', None)
    url = query_params.get('url', None)
    metadata = list()
    pages = [url,]
    while url :
        logger.info("Requesting page: {}".format(url))
        r = requests.get(url,  proxies=proxies)
        if r.status_code != 200:
            logger.error("HTTP Error {} - {}".format(r.status_code, r.reason))
            break
        try:
            load = r.json()
        except JSONDecodeError as e:
            load = r.text
            logger.error("Caught Exception!", exc_info=True)
        metadata.extend(load['notifications'])
        filenames = [get_article_content(x['apiUrl'],api_key,proxies,data_dir) for x in load['notifications']]

        new_link = load['links'][0]['href']
        pages.append(new_link)
        logger.info("Next page link: {}".format(new_link))
        old_url = url.replace("&apiKey={}".format(api_key),"")
        if old_url != new_link:
            url = "{}&apiKey={}".format(new_link,api_key)
            continue
        else:
            return metadata,pages

def get_current_month_begaining():
    
    today=dt.datetime.today()
    current = dt.datetime(today.year,today.month,1)
    #'2019-04-01T02:00:00.000Z'
    current = dt.datetime.strftime(current,"%Y-%m-%d")
    logger.info("{}-T02:00:00.000Z".format(current))
    
    return current

def download_docs(args):
    startTotal = timer()
    ## check folder exists
    if os.path.exists(args.data_dir):
        pass
    else:
        os.mkdir(args.data_dir)
        
    filename = args.cred_path #File with API key
    
    if args.last_update:
        exists = os.path.isfile(os.path.join(config.HISTORICAL_INPUT,'LAST_LINK.txt')) #File with last link saved from previous 
    else:
        exists=False
        
    if exists:
        query_params = get_params(filename, history=exists)
    else:
        query_params = get_params(filename, history=False,since=args.date_since)
    
    logger.info("Query parameters: {}".format(query_params))
    metadata,pages = iterate_through_pages(query_params,args.data_dir)

    # Delete last duplicate entry 
    pages.pop(-1)
    # Save the last available link to the next page (subsequent code runs should start with this link)
    last_link = pages[-1]
    with open(os.path.join(config.HISTORICAL_INPUT,"LAST_LINK.txt"), 'w') as f:
        f.write(last_link.strip())

    # Write out information about this specific code run
    endTotal = timer()
    logger.info('Total runtime: {} seconds.'.format(time.strftime('%H:%M:%S',time.gmtime(endTotal - startTotal))))   
    logger.info("Iterated over {} pages".format(len(pages)))
    logger.info("Last 'next page' link: {}".format(last_link))
    logger.info("Pages list [[REFERENCE]]: {}".format(pages))
    print('Total runtime: {} seconds.'.format(time.strftime('%H:%M:%S',time.gmtime(endTotal - startTotal)))) 
    
def ft_api_args():
    parser = argparse.ArgumentParser()
    current_batch = dt.datetime.strftime(dt.datetime.today(),"%Y-%m")
    parser.add_argument('-dd', '--data_dir', action='store', dest='data_dir',                
                        default=os.path.join(config.JSON_RAW,"{}".format(current_batch)))
    parser.add_argument('-ds', '--date_since', action='store', dest='date_since', default=get_current_month_begaining())
    parser.add_argument('-hs', '--last_update', action='store', dest='last_update', default=False)
    parser.add_argument('-cred', '--cred', action='store', dest='cred_path', default='./credentials_ft.txt')
    args = parser.parse_args()
    return args

#%%
if __name__ == "__main__":

    #global homedir
    #data_dir = args.homedir

    args = ft_api_args()
    
    ## hard code cred_path
    args.cred_path = '/data/News_data_raw/Production/credentials_ft.txt'
    ######
    
    print(args)
    download_docs(args)




