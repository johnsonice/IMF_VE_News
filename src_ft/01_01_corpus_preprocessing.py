"""
Light preprocessing for corpus
"""
import sys
sys.path.insert(0, './libs')
import config
import argparse
import os
import re
import ujson as json
from mp_utils import Mp
from stream import SentStreamer_fast as SentStreamer
import spacy
nlp = spacy.load("en_core_web_lg", disable=['tagger', 'ner', 'parser', 'textcat'])
from spacy.symbols import ORTH, LEMMA, POS, TAG
special_case = [{ORTH: u'__NUMBER__', LEMMA: u'__NUMBER__'}]
nlp.tokenizer.add_special_case(u'__NUMBER__', special_case)


def preprocess(json_article, lemma=True):
    """
    Process entire body of an article, then return after normalizing, lemmatizing
    :param json_article: a dict representing a single article. Must map key('body') -> value(text of article) to work
        properly
    :param lemma: A boolean flag indicating if the article should be lemmatized. Optional. Default = True.
    :return:
    """

    try:
        text = json_article['body']
        
        # Normalize spacing
        text = re.sub("\s+", " ", text)
        text = re.sub("\'+", " ", text)
        
        # Normalize numbers (the fact that a number appears may be important, but not the actual number)
        text = re.sub("([',./]?\d+[',./]?)+", " __NUMBER__ ", text)
        
        # Lemmantize
        if lemma:
            toks = nlp(text)
            toks = [tok.lemma_ for tok in toks if not tok.is_space]
            text = ' '.join(toks)
            
        json_article['body'] = text
        return json_article

    # Alert and proceed if no/null article body
    except (AttributeError, KeyError):
        print('No text in article {}'.format(json_article['an']))
        return False


def punct_space(token):
    """
    Return bool of if token is a punctuation or space
    :param token: A nlp token representation of a word
    :return: True if token is a punctuation or space, False otherwise
    """
    return token.is_punct or token.is_space


class args_class(object):
    def __init__(self, in_dir,out_dir,verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.verbose = verbose


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-raw', '--raw_data_path', help='directory to pull raw data from',
                        default=config.RAW_DATA_PATH)
    parser.add_argument('-out', '--out_directory', help='directory to write the pre-processed corpus to',
                        default=config.JSON_LEMMA)
    parser.add_argument('-v', '--verbose', action='store_true', help='print optional outputs to console')
    args = parser.parse_args()

    # Temp
    args.verbose = True

    # grab all files
    flist = SentStreamer(args.in_dir).input_files
    fl_len = len(flist)
    print('Total number of files to process {}.\n'.format(fl_len))

    ## process and dump processed files
    def process_jsons(fname,out_dir=args.out_dir):
        try: 
            with open(fname, 'r') as f:
                fj = preprocess(json.loads(f.read()))
            if fj and len(fj['body'])>0:
                outf = os.path.join(args.out_dir, fj['an']+'.json')
                with open(outf, 'w') as _f:
                    _f.write(json.dumps(fj))
                #print('file produced')
        except:
            print(fname)
            
    ## multi process files 
    mp = Mp(flist,process_jsons) 
    res = mp.multi_process_files(workers=2,chunk_size=5000)
    


