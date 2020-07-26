#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:13:57 2019

@author: chuang
"""

import os
import sys
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = '.'
    
sys.path.insert(0,os.path.join(cwd,'./libs'))
sys.path.insert(0,os.path.join(cwd,'..'))

import pandas as pd
import numpy as np
#from crisis_points import country_dict
import argparse
from frequency_utils import signif_change
import infer_config as config 
import infer_utils

#%%

class data_updator():
    def __init__(self,args):
        self.args = args
        print('Frequency generator initialized...')
    
    @staticmethod
    def append_update_data(new_ts_path,old_ts_path,update=True):
        
        new_df = pd.read_csv(new_ts_path,index_col=0)
        old_df = pd.read_csv(old_ts_path,index_col=0)
        
        if update:
            new_start = new_df.index[0]
            try:
                old_end_index = list(old_df.index).index(new_start)
                old_df = old_df[:old_end_index]
                updated_df = old_df.append(new_df,sort=True)
                print('some old data has been updated with new data points for past monthes')
            except:
                updated_df = old_df.append(new_df,sort=True)
                print('new data has been appended')
        else:
            raise Exception('only update method has been implemented, please pass in update = True .')
            
        return updated_df

    def calculate_signal(self,df):
        preds = list(signif_change(df, 
                           self.args.window, 
                           period='month',
                           direction='incr',
                           z_thresh=self.args.z_thresh).index) 
        pred_df = pd.DataFrame(np.ones(len(preds)),index=preds,columns=[df.name+'_pred_crisis'])
        df = pred_df.join(df,how='right')[df.name+'_pred_crisis']
        return df
    
    def cal_merge_all_signals(self,df_all):
        var_names = [x for x in list(df_all.columns) if '_pred_crisis' not in x]
        for vn in var_names:
            df_signal = self.calculate_signal(df_all[vn])
            df_all[vn+ '_pred_crisis'] = df_signal
            
        return df_all
    
    def export_all_updated_data(self):
        for c in self.args.countries:
            new_ts_path = os.path.join(self.args.new_ts_folder,"agg_{}_month_z{}_time_series.csv".format(c,self.args.z_thresh))
            old_ts_path = os.path.join(self.args.old_ts_folder,"agg_{}_month_z2.1_time_series.csv".format(c))
            
            country_res = self.append_update_data(new_ts_path,old_ts_path)
            new_res = self.cal_merge_all_signals(country_res)
            out_file = os.path.join(self.args.out_dir,"agg_{}_month_z{}_time_series.csv".format(c,self.args.z_thresh))
            new_res.to_csv(out_file,encoding='utf-8')
        
        print('all file updated at {}'.format(self.args.out_dir))

def get_dm_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--countries', nargs='+', 
                        help='countries to get freq for',
                        default=config.countries)
    parser.add_argument('-nts', '--ts_folder', action='store', dest='new_ts_folder', 
                        default=config.CURRENT_TS_PS)
    parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', 
                        default=config.CURRENT_TS_PS)
    parser.add_argument('-ots', '--old_ts_folder', action='store', dest='old_ts_folder', 
                        default=config.HISTORICAL_TS_PS)
    parser.add_argument('-w', '--window', action='store', dest='window', type=int,
                        default=config.smooth_window_size)
    parser.add_argument('-z', '--z_thresh', action='store', dest='z_thresh', type=int,
                        default=config.z_thresh)
    args = parser.parse_args()
    return args

#%%

if __name__ == '__main__':

    args = get_dm_args(config)
    du = data_updator(args)
    du.export_all_updated_data()