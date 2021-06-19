import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore ,get_input_words_weights,get_country_stats, \
    get_preds_from_pd, get_eval_stats
import pandas as pd
import numpy as np
import os
import config
import glob

def add_means(df, include_subgroups=True):
    overall_pos = df.columns[17::2]
    overall_neg = df.columns[18::2]
    df['all_pos_mean'] = df[overall_pos].T.mean()
    df['all_neg_mean'] = df[overall_neg].T.mean()

    if include_subgroups:
        fed_pos = df.columns[17:28:2]
        fed_neg = df.columns[18:29:2]
        df['fed_pos_mean'] = df[fed_pos].T.mean()
        df['fed_neg_mean'] = df[fed_neg].T.mean()

        w2v_pos = df.columns[29:40:2]
        w2v_neg = df.columns[30:41:2]
        df['w2v_pos_mean'] = df[w2v_pos].T.mean()
        df['w2v_neg_mean'] = df[w2v_neg].T.mean()

        w2vRef_pos = df.columns[41::2]
        w2vRef_neg = df.columns[42::2]
        df['w2vRef_pos_mean'] = df[w2vRef_pos].T.mean()
        df['w2vRef_neg_mean'] = df[w2vRef_neg].T.mean()


def add_medians(df, include_subgroups = True):
    overall_pos = df.columns[17::2]
    overall_neg = df.columns[18::2]
    df['all_pos_med'] = df[overall_pos].T.median()
    df['all_neg_med'] = df[overall_neg].T.median()

    if include_subgroups:
        fed_pos = df.columns[17:28:2]
        fed_neg = df.columns[18:29:2]
        df['fed_pos_med'] = df[fed_pos].T.median()
        df['fed_neg_med'] = df[fed_neg].T.median()

        w2v_pos = df.columns[29:40:2]
        w2v_neg = df.columns[30:41:2]
        df['w2v_pos_med'] = df[w2v_pos].T.median()
        df['w2v_neg_med'] = df[w2v_neg].T.median()

        w2vRef_pos = df.columns[41::2]
        w2vRef_neg = df.columns[42::2]
        df['w2vRef_pos_med'] = df[w2vRef_pos].T.median()
        df['w2vRef_neg_med'] = df[w2vRef_neg].T.median()

if __name__ == "__main__":

    in_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_mean2')
    out_dir = os.path.join(config.EVAL_WordDefs, 'final_sent_mean2_mm')
    all_files = glob.glob(os.path.join(in_dir, '*'))
    for file in all_files:
        in_df = pd.read_csv(file)
        add_means(in_df)
        add_medians(in_df)
        out_name = os.path.join(out_dir, file.split('/')[-1])
        in_df.to_csv(out_name)
    