"""
collate info for documents to speed things up downstream
"""
import sys
sys.path.insert(0,'./libs')
import config
import argparse
import pandas as pd
from datetime import datetime as dt
import ujson as json
from glob import glob
from crisis_points import crisis_points
from frequency_utils import list_crisis_docs
import os
from stream import MetaStreamer_fast as MetaStreamer

#%%
#def time_index(docs, lang=None, verbose=False):
#    doc_details = {}
#    tot = len(docs)
#    for i, doc in enumerate(docs):
#        if verbose:
#            print('\r{} of {} processed'.format(i, tot), end='')
#        with open(doc, 'r', encoding='utf-8') as f:
#            art = json.loads(f.read())
#            try:
#                if lang:
#                    if art['language_code'] != lang:
#                        continue
#                date = pd.to_datetime(dt.fromtimestamp(art['publication_date'] / 1e3))
#                doc_details[art['an']] = {'date': date}
#            except Exception as e:
#                print(art['an'] + ': ' + e.characters_written)region_matched
#    data = pd.DataFrame(doc_details).T
#    return data
## multi processed json input
def time_index(docs, lang=None, verbose=False,date_format='DNA'):
    doc_details = {}
    tot = len(docs)
    print('Convert dates....')
    for i, doc in enumerate(docs):
        if verbose:
            print('\r{} of {} processed'.format(i, tot), end='',flush=True)
        try:
            if date_format.strip().lower() == 'ft':
                date = pd.to_datetime(dt.strptime(doc['publication_date'],'%Y-%m-%d'))
            else:
                date = pd.to_datetime(dt.fromtimestamp(doc['publication_date'] / 1e3))
            doc_details[doc['an']] = {'date': date}
        except Exception as e:
            print(doc['an'] + ': ' + str(e))
            
    data = pd.DataFrame(doc_details).T
    return data
#%%

def period_info(doc_deets):
    dates = pd.DatetimeIndex(doc_deets['date'])
    doc_deets['week'] = dates.to_period('W')
    doc_deets['month'] = dates.to_period('M')
    doc_deets['quarter'] = dates.to_period('Q')
    return doc_deets


def label_crisis(data, path, verbose=False, period='crisis'):
    data['crisis'] = 0
    crisis = []
    for country in crisis_points.keys():
        if verbose:
            print("\nworking on {}...".format(country))
        crisis_docs = list_crisis_docs(country, path,doc_data=data, period=period)
        crisis_ids = [os.path.basename(doc).replace(".json", '') for doc in crisis_docs]
        crisis += crisis_ids
    data.loc[data.index.isin(crisis), 'crisis'] = 1
    return data
 
class args_class(object):
    def __init__(self, in_dir,out_dir,period='crisis',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.period=period
        self.verbose = verbose
        
#%%
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
        parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
        parser.add_argument('-p', '--period', action='store', dest='period', default='crisis')
        parser.add_argument('-v', '--verbose', action='store', dest='verbose', default=True)
        args = parser.parse_args()
    except:
        args = args_class(config.JSON_LEMMA,config.DOC_META, verbose = True)
    
    streamer = MetaStreamer(args.in_dir, language='en',verbose=args.verbose)  
    deets = time_index(streamer.multi_process_files(workers=31,chunk_size=5000), lang='en', verbose=False,date_format='FT')
    deets = period_info(deets)
    deets = label_crisis(deets, path = args.in_dir, verbose=args.verbose, period=args.period)
    deets.to_pickle(os.path.join(args.out_dir, 'doc_details_{}.pkl'.format(args.period)))
 
