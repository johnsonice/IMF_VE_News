#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 05:38:37 2020

@author: chuang
"""

import os,json
import config
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Inference_model(object):
    def __init__(self,trained_weights_path,label_map_path,load_mode='full',device='cpu'):
        self.model_path = trained_weights_path
        self.load_mode = load_mode
        self.device = device
        self.label_map = self.load_label_map(label_map_path)
        if load_mode == 'full':
            self.model = torch.load(trained_weights_path)
            self.model.eval()
    
    @staticmethod
    def load_label_map(map_path):
        with open(map_path,'r') as fp:
            data = json.load(fp)
        return data
    
    @staticmethod
    def prepare_input(X):
        data_iter = torch.utils.data.DataLoader(X, batch_size=16,shuffle=False)
        return data_iter
    
    def predict_from_emb(self,inputs):
        predicted_label = []
        data_iter = self.prepare_input(inputs)
        with torch.no_grad():
            for batch_idx, X in enumerate(data_iter): 
                i = X.to(self.device)#.float()

                outputs = self.model(i) 
                _, predicted = torch.max(outputs.data, 1)
                predicted_label.extend(predicted.tolist())
        
        predicted_label_name = [self.label_map[str(i)] for i in predicted_label]
        return predicted_label,predicted_label_name
    
    def predict_from_text(self,inputs):
        
        return None

def get_precision(tp,fp):
    try:
        return tp/(tp+fp)
    except ZeroDivisionError:
        return np.nan

def get_recall(tp,fn):
    try:
        return tp/(tp+fn)
    except ZeroDivisionError:
        return np.nan
    
def get_fscore(tp,fp,fn,beta):
    try:
        return ((1+beta**2)*tp)/((1+beta**2)*tp + (beta**2*fn)+fp)
    except ZeroDivisionError:
        return np.nan
    
def get_eval_matrics_one_country():
    """
    get evaluation metrics by couuuntry
    input is a pandas df with time;crisis
    """
    
    
    return None