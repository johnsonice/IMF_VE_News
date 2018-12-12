"""
Create ngram phraser obj for input corpus
Note: The phraser objects created here are what get passed to the streamer using the 'phraser' parameter.
"""
import sys
sys.path.insert(0,'./libs')
import gensim
import os
from stream import SentStreamer_fast as SentStreamer
import argparse
import config 

#%%
def ngram_phraser(n, corpus, scoring, thresh, min_count, common_terms, language=None,verbose=False):
    """
    Creates a Gensim phrasegram model
    :param n: (int) ngram rank
    :param corpus: (str) dir where data is housed
    :param scoring: (str) 'npmi' or 'default'.
    :param thresh: (float) threshold for scoring function
    :param min_count: (int) min number of times a token must appear in corpus in order to be considered
    :param common_terms: (list of str) tokens whose presence between two words won’t prevent bigram detection. 
        It allows the detection of expressions like “bank of america” or “eye of the beholder”.
    :return: (gensim.models.phrases.Phraser) Phraser object of ngram rank n.
    """
    assert n >= 2

    # use sent_stream generator to feed data to the phraser
    streamer = SentStreamer(corpus, language=language,stopwords=[], verbose=verbose)
    if len(streamer.input_files) > config.SAMPLE_LIMIT:
        streamer.input_files = streamer.input_files[-config.SAMPLE_LIMIT:]
        print("number of input files is too large: only load last {}".format(config.SAMPLE_LIMIT))
        
    if n == 2:
        print('Working on {}grams...'.format(n))
        phrase_model = gensim.models.Phrases(streamer.multi_process_files(), scoring=scoring, min_count=min_count, threshold=thresh,
                                             common_terms=common_terms)
    else:
        prev_phraser = ngram_phraser(n - 1, corpus, scoring, thresh, min_count, common_terms, language=language)
        print('Working on {}grams...'.format(n))
        phrase_model = gensim.models.Phrases(prev_phraser[streamer.multi_process_files()], scoring=scoring, min_count=min_count,
                                             threshold=thresh + thresh * 0.5,
                                             common_terms=common_terms)

    return gensim.models.phrases.Phraser(phrase_model)

class args_class(object):
    def __init__(self, in_dir,out_dir,scoring='default',thresh = 10, min_count=20, common_terms=(),n_rank=2,lang='en',verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.scoring= scoring
        self.thresh= thresh
        self.min_count= min_count
        self.common_terms = common_terms
        self.n_rank=n_rank
        self.lang=lang
        self.verbose = verbose
        
#%%
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
        parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
        parser.add_argument('-s', '--scoring', action='store', dest='scoring', default='default')
        parser.add_argument('-mc', '--min_count', action='store', dest='min_count', default=20)
        parser.add_argument('-t', '--threshold', action='store', dest='thresh', default=10)
        parser.add_argument('-ct', '--common_terms', action='store', dest='common_terms', default=())
        parser.add_argument('-n', '--n_rank', action='store', dest='n_rank', default=2)
        parser.add_argument('-lang', '--language', action='store', dest='lang', default='en')
        parser.add_argument('-v', '--verbose', action='store', dest='verbose', default=False)
        args = parser.parse_args()
    except:
        args = args_class(in_dir='/data/News_data_raw/FT_WD/json_lemma',out_dir='/data/News_data_raw/FT_WD/models/ngrams',verbose=True)
 
    # construct model
    ngrams = ngram_phraser(args.n_rank, args.in_dir, args.scoring, args.thresh, args.min_count, 
                           args.common_terms, args.lang,verbose=args.verbose)

    # save ngram model
    stop_flag = '_NOSTOP' if not args.common_terms else ''
    ngram_model_name = '{}grams_{}_{}_{}{}'.format(args.n_rank, args.scoring, args.thresh, 
                                                   args.min_count, stop_flag)
    save_path = os.path.join(args.out_dir, ngram_model_name)
    ngrams.save(save_path)
