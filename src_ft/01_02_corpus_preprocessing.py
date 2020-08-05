"""
Light preprocessing for corpus
"""
import sys
sys.path.insert(0,'./libs')
import config
import os
import ujson as json
from mp_utils import Mp
from stream import SentStreamer_fast as SentStreamer
from spacy.lang.en.stop_words import STOP_WORDS as stops
bad_char = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?',
            '@', '[', '\\', ']', '^', '`', '{', '|', '}', '~', 'NUMBER', '=', '-', '«', '.']


def deep_clean(token):
    """
    Remove tiny and very long words, stopwords, punctuations and numbers
    :param token: 'str' A string representations of a word
    :return: True if the token to be kept, False otherwise
    """

    # Remove tiny words
    if len(token) <= 3:
        return False

    # Remove extremely long words
    if len(token) > 20:
        return False

    # Remove stopwords
    if token in stops:
        return False

    # Remove numbers, punctuation
    for b in bad_char:
        if b in token:
            return False

    # Token is to be kept
    return True


def preprocess(json_article):
    """
    Process entire body of an article, then save after eliminating unwanted words, signs
    :param json_article: a dict representing a single article. Must map key('body') -> value(text of article) to work
        properly
    :return: The article with body cleaned / False if the article body does not exist
    """

    # Try to read the file
    try:
        text = json_article['body']
        tokens = text.split(' ')
        tokens = [t for t in tokens if deep_clean(t)]  # Keep only desirable tokens
        
        text = ' '.join(tokens)
        json_article['body'] = text
        return json_article

    # If no body in article, print alert, return False
    except (KeyError, AttributeError):
        print('no text in article {}'.format(json_article['an']))
        return False        


class args_class(object):
    """
    A class representing the modulable configuration of the program
    """

    def __init__(self, in_dir,out_dir,verbose=True):
        self.in_dir = in_dir  # Dictionary top read files from
        self.out_dir = out_dir  # Dictionary to write processed files to
        self.verbose = verbose  # Boolean flag to control optional print alerts


if __name__ == '__main__':

    # TODO make modable through bash
    args = args_class(config.JSON_LEMMA, config.JSON_LEMMA_SMALL, verbose=True)
        
    # grab all files
    flist = SentStreamer(args.in_dir).input_files
    fl_len = len(flist)
    print('Total number of files to process {}.\n'.format(fl_len))

    # process and dump pre-processed files
    def process_jsons(fname, out_dir=args.out_dir):
        # Try to pre-process file at fname
        try:
            # Pre-process the file
            with open(fname, 'r') as f:
                fj = preprocess(json.loads(f.read()))
            # IF article has normal body, write the article to file
            if fj and len(fj['body']) > 0:
                outf = os.path.join(out_dir, fj['an']+'.json')
                with open(outf, 'w') as _f:
                    _f.write(json.dumps(fj))
        # If not found, alert and proceed
        except FileNotFoundError:
            print("File not found: {}".format(fname))
            
    # multi process files - consider processing power and memory when determining workers, chunk size
    mp = Mp(flist, process_jsons)
    res = mp.multi_process_files(workers=20, chunk_size=10000)  # File-specific and machine-specific optimization
    


