"""
Create dataframe of topic results for given model
"""
import gensim
import os
from glob import glob
import json
import pandas as pd
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir')
    args = parser.parse_args()

    res_files = glob(args.in_dir + '/*.json')
    analytics = []
    for i, res in enumerate(res_files):
        print('\rworking on {}'.format(res), end='')
        with open(res, 'r') as f:
            data = json.loads(f.read())
        info = data['overall_results']
        info['terms'] = ' | '.join(data['terms'])
        analytics.append(pd.Series(data['overall_results'], name='topic_{}'.format(data['topic'])))
    ana_frame = pd.DataFrame(analytics)
    ana_frame = ana_frame.sort_values(by=['fscore','recall'], ascending=False)
    ana_frame.to_csv(os.path.join(args.in_dir, 'summary_{}.csv'.format(os.path.basename(args.in_dir))))



