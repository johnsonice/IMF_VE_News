"""
frequency_country_specific_freqs.py

Description: retrieve and save country-specific word frequencies for each supplied country. The word 
freq data for each country will only be based on articles which either mention the country name in the
title or abstract, or which are labeled with the region code corresponding to that particular country. 

usage: python3 frequency_country_specific_freqs.py
NOTE: can be done for as many countries at a time as you want.
"""
import logging
import os
import sys
sys.path.insert(0,'./libs')
import pandas as pd
from collections import defaultdict
from stream import DocStreamer_fast
from crisis_points import country_dict
from gensim.models.keyedvectors import KeyedVectors
#from region_mapping import region
import argparse
import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# create a file handler
handler = logging.FileHandler('converting.log')
handler.setLevel(logging.WARNING)
# create logging format 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add handlers to logger
logger.addHandler(handler)

def data_setup(doc_deets, period):
    time_df = pd.read_pickle(doc_deets)
    uniq_periods = set(time_df[period])
    time_df = time_df[time_df['country_n'] > 0]  ## filter only docs with countries

    return time_df, uniq_periods

def country_period_filter(time_df,country,period):
    
    time_df['filter_country'] = time_df['country'].apply(lambda c: country in c)
    df = time_df['data_path'][(time_df['filter_country'] == True)&(time_df[args.period] == period)]
    
    return df.tolist()


def get_country_freqs(countries, period_choice, time_df, uniq_periods,outdir,phraser,filter_dict=None):

    # Get frequency data for each country supplied
    print("\nCounting Word Freqs...")
    for country in countries:
        logger.info("\nWorking on {}".format(country))
        freqs = {}
        #for i, (period, doc_list) in enumerate(period_dict.items()):
        for i, period in enumerate(uniq_periods):
            print("\r\tworking on period {} of {}...".format(i, len(uniq_periods)), end=' ')
            p_freqs = defaultdict(int)
            doc_list = country_period_filter(time_df,country,period)
            doc_list = [os.path.join(config.JSON_LEMMA,os.path.basename(p)) for p in doc_list]
            print("Files to process: {}".format(len(doc_list)))
#            streamer = DocStreamer_fast(doc_list, language='en', regions=[region[country]], region_inclusive=True,
#                                    title_filter=[country],
#                                    phraser=phraser,lemmatize=False).multi_process_files(workers=int(os.cpu_count()/2),chunk_size = 500)
            streamer = DocStreamer_fast(doc_list, language='en', phraser=phraser,
                                        stopwords=[], lemmatize=False).multi_process_files(workers=15,chunk_size = 50)
            # count
            for doc in streamer:
                for token in doc:
                    if filter_dict is None:
                        p_freqs[token] += 1
                    else:
                        if token in filter_dict:
                            #print(token)
                            p_freqs[token] += 1

            # Normalize
            total_tokens = sum(p_freqs.values())
            per_thousand = total_tokens / 1000
            p_freqs = {k: v / per_thousand for k, v in p_freqs.items()}

            # Add to master dict
            freqs[period] = p_freqs

        # Fill NAs as 0
        freqs_df = pd.DataFrame(freqs)#.fillna(0)
        # make sure columns are in ascending order
        freqs_df = freqs_df[freqs_df.columns.sort_values()] 

        ##
        # for a test 
        ##
#        freqs_df['sum'] = freqs_df.sum(axis=1)
#        freqs_df  = freqs_df.sort_values(by='sum',ascending=False).head(200000)
#        freqs_df.drop(columns=['sum'],inplace=True)
        #return freqs_df
        
        # write csv
        #try:
        out_csv = os.path.join(outdir, '{}_{}_word_freqs.csv'.format(country, period_choice))
        freqs_df.to_csv(out_csv)
        print('Country: {} saved to csv at {}.'.format(country, out_csv))
        del freqs_df
        #except:
        #    logger.warning("Problem saving country: {}. Skipped for now.".format(country))
        #    del freqs_df
            
#        # write pkl
##        try:
##            out_pkl = os.path.join(outdir, '{}_{}_word_freqs.pkl'.format(country, period_choice))
##            freqs_df.to_pickle(out_pkl)
##        except Exception as e:
##            print("Problem: while saving pickle. try to load data from CSV")
##            raise Exception(e)
#        
#            return freqs_df

class args_class(object):
    def __init__(self, corpus=config.JSON_LEMMA,doc_deets=config.AUG_DOC_META_FILE,
                 countries=country_dict.keys(),period=config.COUNTRY_FREQ_PERIOD,out_dir=config.FREQUENCY,
                 phraser=config.PHRASER):
        self.corpus = corpus
        self.doc_deets = doc_deets
        self.countries = countries
        self.period = period
        self.out_dir = out_dir
        self.phraser=phraser
        
#%%
if __name__ == '__main__':

    class_type_setups = config.class_type_setups




    parser = argparse.ArgumentParser()
    countries = config.countries

    parser.add_argument('-c', '--countries', nargs='+', help='countries to get freq for',
                        default=countries)
    parser.add_argument('-corp', '--corpus', action='store', dest='corpus',
                        default=config.JSON_LEMMA)
    parser.add_argument('-deets', '--doc_details', action='store', dest='doc_deets',
                        default=config.AUG_DOC_META_FILE)
    parser.add_argument('-p', '--period', action='store', dest='period',
                        default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-s', '--save_dir', action='store', dest='out_dir',
                        default=config.FREQUENCY)
    parser.add_argument('-ph', '--phraser', action='store', dest='phraser',
                        default=config.PHRASER)
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', default=config.W2V)
    parser.add_argument('-f', '--wv_filter', action='store', dest='wv_filter', default='TRUE')
    args = parser.parse_args()
#    except:
#        args = args_class(corpus=config.JSON_LEMMA,doc_deets=config.AUG_DOC_META_FILE,
#                          out_dir=config.FREQUENCY,period=config.COUNTRY_FREQ_PERIOD,
#                          phraser=config.PHRASER,countries=config.countries)
        #%%

    ## load dictionary
    if args.wv_filter == 'TRUE':
        vecs = KeyedVectors.load(args.wv_path)
        wv_keys = vecs.wv.vocab.keys()
        logger.info('use wv_filder, max length: {}'.format(len(wv_keys)))
        del vecs  ## clear memory
    else:
        wv_keys = None

    # TODO rethink framework
    experimental = config.experimenting
    mode = config.experiment_mode
    if experimental:

        if mode == "country_classification":
            for setup in class_type_setups:
                class_type = setup[0]
                args.doc_deets = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format(class_type))
                args.out_dir = os.path.join(config.FREQUENCY, class_type)
                time_df, uniq_periods = data_setup(args.doc_deets, args.period)

                get_country_freqs(args.countries, args.period, time_df, uniq_periods, args.out_dir, args.phraser,
                                  filter_dict=wv_keys)
        elif mode == "topiccing_discrimination":
            topic_aug_folder = config.topiccing_aug_meta
            topic_f2_thresholds = config.topic_f2_thresholds
            document_topic_thresholds = config.document_topic_min_levels

            debug = True
            if debug:
                topic_f2_thresholds = [('top', 10), .5]
                #document_topic_thresholds = [("top", 2), .5]
                document_topic_thresholds = [.5]
            class_type = "Min1_AllCountry"

            for f2_thresh in topic_f2_thresholds:
                if type(f2_thresh) is tuple:
                    f2_thresh = '{}_{}'.format(f2_thresh[0], f2_thresh[1])
                else:
                    f2_thresh = str(f2_thresh)

                for doc_thresh in document_topic_thresholds:
                    if type(doc_thresh) is tuple:
                        doc_thresh = '{}_{}'.format(doc_thresh[0], doc_thresh[1])
                    else:
                        doc_thresh = str(doc_thresh)

                    args.doc_deets = os.path.join(topic_aug_folder,
                                                  'doc_meta_aug_threshold_{}_setup_{}_docMinLevel_{}.pkl'.format(
                                                      f2_thresh, class_type, doc_thresh))
                    args.out_dir = os.path.join(config.topiccing_frequency, class_type, f2_thresh, doc_thresh)
                    if config.just_five:
                        args.out_dir = os.path.join(args.out_dir, "j5_countries")

                    time_df, uniq_periods = data_setup(args.doc_deets, args.period)

                    get_country_freqs(args.countries, args.period, time_df, uniq_periods, args.out_dir, args.phraser,
                                      filter_dict=wv_keys)

    else:
        # Data setup
        time_df, uniq_periods = data_setup(args.doc_deets, args.period)

        get_country_freqs(args.countries, args.period, time_df, uniq_periods, args.out_dir, args.phraser,
                          filter_dict=wv_keys)
