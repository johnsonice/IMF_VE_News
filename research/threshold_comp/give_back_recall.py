import numpy as np
import os
import pandas as pd


def get_recall(tp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return np.nan


def get_precision(tp, fp):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return np.nan

# Get files
files = [f for f in os.listdir('.') if os.path.isfile(f)]
files.remove('give_back_recall.py')


# Fix float imprecision drift
for f in files:
    print(f)
    load_df = pd.read_csv(f)
    for ind in load_df.index:
        loc_thresh = load_df.loc[ind, 'threshold']
        rd_thresh = round(loc_thresh, 3)
        load_df.loc[ind, 'threshold'] = rd_thresh
    os.remove(f)
    load_df.to_csv(f, index=False)

# Give back recalls, 1-sens, prec
thresholds = [0, 1.282, 1.44, 1.645, 1.96, 2.576, 99]
countries = ['argentina', 'bolivia', 'brazil', 'chile', 'colombia']
for f in files:
    load_df = pd.read_csv(f, index_col=['threshold', 'country'])
    #print(load_df)

    load_df['recall'] = np.nan
    load_df['precision'] = np.nan
    load_df['one_minus_sens'] = np.nan
    for thresh in thresholds:
        for country in countries:
            cur_loc = (thresh, country)
            tp = load_df.loc[cur_loc, 'tp']
            fn = load_df.loc[cur_loc, 'fn']
            fp = load_df.loc[cur_loc, 'fp']
            sens = load_df.loc[cur_loc, 'sensitivity']
            load_df.loc[cur_loc, 'recall'] = get_recall(tp, fn)
            load_df.loc[cur_loc, 'precision'] = get_precision(tp, fp)
            load_df.loc[cur_loc, 'one_minus_sens'] = 1 - sens
    os.remove(f)
    load_df.to_csv(f)

