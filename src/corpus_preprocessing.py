"""
Light preprocessing for corpus
"""
import os
from glob import glob
import re
import ujson as json
import argparse


def preprocess(json_article):
    try:
        text = json_article['body']
        # Normalize capitalization
        text = text.lower()

        # Normalize spacing
        text = re.sub("\s+", " ", text)

        # Normalize numbers (the fact that a number appears may be important, but not the actual number)
        text = re.sub("(\d+[,./]?)+", "<<NUMBER>>", text)

        json_article['body'] = text
        return json_article
    except Exception:
        print('no text in article {}'.format(json_article['an']))
        return False

class args_class(object):
    def __init__(self, in_dir,out_dir,verbose=True):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.verbose = verbose

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
        parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
        parser.add_argument('-o', '--verbose', action='store', dest='verbose', default=False)
        args = parser.parse_args()
    except:
        ## give some default arguments
        args = args_class('../cleaned_small','../data/processed_json', verbose = True)
        
    flist = glob(args.in_dir + '/*.json')
    fl_len = len(flist)
    for idx,fname in enumerate(flist):
        with open(fname, 'r') as f:
            if args.verbose:
                print('\rProcessing number {}/{}'.format(idx,fl_len),end='',flush=True)
                
            fj = preprocess(json.loads(f.read()))

        if fj:
            outf = os.path.join(args.out_dir, os.path.basename(fname))
            with open(outf, 'w') as f:
                f.write(json.dumps(fj))
