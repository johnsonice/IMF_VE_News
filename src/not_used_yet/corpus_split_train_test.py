"""
Create random test/train split from corpus
"""
import random
import os
from glob import glob
import argparse


def sym_link_files(file_list, path, verbose=False):
    for i, f in enumerate(file_list):
        dest = os.path.join(path, os.path.basename(f))
        try:
            os.symlink(f, dest)
        except:
            pass
        
        if verbose:
            if i % 100 == 0:
                print('\r{} of {} test docs copied'.format(i, len(test_files)), end='')

class args_class(object):
    def __init__(self, in_dir,test_dir,train_dir,test_ratio=0.2,verbose=False):
        self.in_dir = in_dir
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.test_ratio = test_ratio
        self.verbose = verbose


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', reguired=True)
        parser.add_argument('-test', '--test_dir', action='store', dest='test_dir', default='../test')
        parser.add_argument('-train', '--train_dir', action='store', dest='train_dir', default='../train')
        parser.add_argument('-ratio', '--test_ratio', action='store', dest='test_ratio', default=0.2)
        parser.add_argument('-v', '--verbose', action='store', dest='verbose', default=False)
        args = parser.parse_args()
    except:
        args = args_class(in_dir = '../data/processed_json',test_dir='../data/test_dir',train_dir='../data/train_dir',verbose=True)

    # Grab a shuffled list of all docs in input_dir
    docs = glob(args.in_dir + '/*.json')
    random.shuffle(docs)

    # Split into test, train based on the test ratio provided
    cutoff = round(len(docs)*args.test_ratio)
    test_files, train_files = docs[:cutoff], docs[cutoff:]

    # Create symlinks in test_dir for test files
    sym_link_files(test_files, args.test_dir, args.verbose)
    sym_link_files(train_files, args.train_dir, args.verbose)
