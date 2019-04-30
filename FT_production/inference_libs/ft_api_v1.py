
import argparse
import requests
import os, sys, csv, re
from timeit import default_timer as timer
import datetime as dt
import json
import logging

# Set logging file name and logging parameters
logname = "FT_Log_{}.log".format(str(dt.datetime.today().strftime("%m.%d.%Y")))
logging.basicConfig(filename=logname,
                    format='%(asctime)s : %(levelname)s - %(message)s',
                    filemode='a',
                    level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#########################################################################################
def get_params(filename, history):
    """
    Read file with API key [credentials_ft.txt] and the last saved page link [LAST_LINK.txt]* if available.
    If last saved page link is unavailable, use default url parameters with start date 2019-03-15.
    Return dictionary with query parameters.
    """
    with open(filename, 'r') as f:
        key = f.readline()
    api_key = key.strip()
    if history:
        with open('./LAST_LINK.txt','r') as l:
            url1 = l.readline()
        main_url = url1.strip()
        timestamp = ""
        url_query = "{}&apiKey={}".format(main_url, api_key) 
    else:
        main_url = "https://api.ft.com/content/notifications?"
        timestamp = '2019-03-15T02:00:00.000Z'
        url_query = "{}since={}&apiKey={}".format(main_url,timestamp, api_key)
    query_params = {
        'proxies':{'http': "socks5://intsquid:8000",'https': "socks5://intsquid:8000"},
        'base_url':main_url,
        'api_key': api_key,
        'start_date': timestamp,
        'url': url_query
        }
     
    return query_params

def generate_file(filename,data):
    """Save content of the article and its metadata in json format to the homedir specified in argparse"""
    with open(os.path.join(homedir,filename), 'w+') as outfile:
        outfile.write(json.dumps(data))#,indent=4, sort_keys=True))

def get_article_content(link,api_key,proxies):
    """
    Get content of specified article. In case the content is unavailable [ERROR 403], log the error and article link.
    Continue iteration over article links on that page. There should be 200 links per page, unless the page is the latest one
    Returns name of the saved file or None
    """
    link = link+"?apiKey={}".format(api_key)
    logger.info("Article link: {}".format(link))
    r = requests.get(link, proxies=proxies)
    if r.status_code == 403:
        logger.warning('Error with status code {}. Article is not available at the moment.'.format(r.status_code))
        return None
    elif r.status_code != 200:
        logger.warning('[?] Error: [HTTP {}]: Content: {}'.format(r.status_code, r.content))
        return None
    else:
        data = r.json()
        name_start = data['id'].split('thing/')[1]
        name_end = data['firstPublishedDate'].split('T')[0]
        file_name = name_start+"_"+name_end+".json"
        logger.info("Generated: {} ".format(file_name))
        generate_file(file_name,data)
        return file_name

def iterate_through_pages(query_params):
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
        filenames = [get_article_content(x['apiUrl'],api_key,proxies) for x in load['notifications']]

        new_link = load['links'][0]['href']
        pages.append(new_link)
        logger.info("Next page link: {}".format(new_link))
        old_url = url.replace("&apiKey={}".format(api_key),"")
        if old_url != new_link:
            url = "{}&apiKey={}".format(new_link,api_key)
            continue
        else:
            return metadata,pages

def main(homedir):
    startTotal = timer()
    filename = "U:\\credentials_ft.txt" #File with API key
    exists = os.path.isfile('./LAST_LINK.txt') #File with last link saved from previous run
    if exists:
        query_params = get_params(filename, history=exists)
    else:
        query_params = get_params(filename, history=False)
    
    logger.info("Query parameters: {}".format(query_params))
    metadata,pages = iterate_through_pages(query_params)

    # Delete last duplicate entry 
    pages.pop(-1)
    # Save the last available link to the next page (subsequent code runs should start with this link)
    last_link = pages[-1]
    with open("./LAST_LINK.txt", 'w') as f:
        f.write(last_link.strip())

    # Write out information about this specific code run
    endTotal = timer()
    logger.info('Total runtime: {} seconds.'.format(endTotal - startTotal))   
    logger.info("Iterated over {} pages".format(len(pages)))
    logger.info("Last 'next page' link: {}".format(last_link))
    logger.info("Pages list [[REFERENCE]]: {}".format(pages))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--homedir', action='store', dest='homedir', default="U:\\files_ft\\")
    args = parser.parse_args()
    global homedir
    homedir = args.homedir
    main(homedir)


