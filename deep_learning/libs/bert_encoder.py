#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:17:54 2020

@author: chuang
"""

## baseline model 
import torch
from transformers import BertTokenizer,BertModel
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

class baseline_model(object):
    def __init__(self, pretrained_weights_path,tokenizer_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.model = BertModel.from_pretrained(pretrained_weights_path)
        
    def convert2input_id(self,input_sentences,max_length=20):
        if isinstance(input_sentences,str):
            input_ids = torch.tensor([self.tokenizer.encode(input_sentences, 
                                           add_special_tokens=True)
                                      ])
            inputs = {'input_ids':input_ids}
        elif isinstance(input_sentences,list):
            batch_encoding = self.tokenizer.batch_encode_plus(input_sentences,
                                                              truncation=True,
                                                              #max_length=max_length,
                                                              pad_to_max_length=True)
            inputs = {k: torch.tensor(batch_encoding[k]) for k in batch_encoding}
        else:
            raise Exception('Input type has to be string or list, please double check')

        return inputs
    
    def tokenize_show(self,input_sentences):
        
        inputs = self.convert2input_id(input_sentences)
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in inputs['input_ids'].tolist()]
        
        return tokens
    
    def bert_encode(self,input_sentences,max_length=512,to_numpy=True,pool=True):
        
        
        inputs = self.convert2input_id(input_sentences,max_length=max_length)
        mask = inputs['attention_mask']
        with torch.no_grad():
            if pool:
                mask_3d = mask.unsqueeze(2).repeat(1,1,768)
                cls_vec = torch.sum(self.model(**inputs)[0] * mask_3d,1)/torch.sum(mask,1).view(-1,1)
                #cls_vec = torch.mean(self.model(input_ids)[0],1)
            else:
                cls_vec = self.model(inputs)[0][:,0,:]
                
            if to_numpy:
                cls_vec = cls_vec.cpu().numpy()
        return cls_vec
    
    
    def convert2input_id2(self,input_sentences,max_length=512):
        if isinstance(input_sentences,str):
            input_ids = [torch.tensor([self.tokenizer.encode(input_sentences, 
                                           add_special_tokens=True,
                                           #max_length=max_length,
                                           truncation=True)
                                      ])]
        elif isinstance(input_sentences,list):
            inputs = [self.tokenizer.encode(i,add_special_tokens=True) for i in input_sentences]
            input_ids = [torch.tensor([i]) for i in  inputs]
        else:
            raise Exception('Input type has to be string or list, please double check')

        return input_ids
    
    def bert_encode2(self,input_sentences,pool=True,max_length=512):
        ## instead of use CSL toke, use pooled vector instead
        input_ids= self.convert2input_id2(input_sentences,max_length=max_length)
        
        with torch.no_grad():
            if pool:
                cls_vec = [torch.mean(self.model(i)[0],1).cpu().numpy().squeeze() for i in input_ids]
            else:
                #res = [self.model(i) for i in input_ids]
                cls_vec = [self.model(i)[0][:,0,:].cpu().numpy().squeeze() for i in input_ids]
                
            cls_vec = np.vstack(cls_vec)
        return cls_vec
    
#%%

if __name__ == "__main__":
    
    model_path = '/data/News_data_raw/FT_WD/models/pre_trained_english/'
    model = baseline_model(model_path)

    test_questions = ['test questions 1','this is the second test question']
    emb = model.bert_encode2(test_questions)
    print(emb.shape)
    
    
    
    
