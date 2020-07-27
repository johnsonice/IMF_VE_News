#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:04:21 2020

@author: chuang
"""

## train simple model 
import os,sys
sys.path.insert(0,'../libs')
from model_baseline import Simple_nn_model,Dynamic_simple_sequencial_model
import config
import torch 
import torch.nn as nn
#from torch.nn import functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_map(df,label_column):
    label2id = {}
    for idx,key in enumerate(df[label_column].unique()):
        label2id[key] = idx
        
    id2label = {i:key for key,i in label2id.items()}
    
    return label2id, id2label

def prepare_torch_training_data(df,x_label,y_label,ratio=0.3):
    """
    prepare data into torch dataset for training
    """
    
    X = df[x_label]
    y = df[y_label]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =ratio,random_state=0)

    train_data = list(zip(X_train,y_train))
    test_data = list(zip(X_test,y_test))

    train_data_iter = torch.utils.data.DataLoader(train_data, batch_size=16,shuffle=True)
    test_data_iter = torch.utils.data.DataLoader(test_data, batch_size=16,shuffle=True)
    #next(iter(train_data_iter))
    
    return train_data_iter,test_data_iter

def train_model(model,optimizer,loss_fn,n_epochs,train_data_iter,
                log_interval=100,loss_scale=1,device="cpu",
                do_eval=False,test_data_iter=None,
                save_criterion=None,warmup_n_epoch=10,
                save_model_path=None):
        
    model.train()
    losses = []
    total_loss = 0
    best_save_criterion = None
    for i in range(n_epochs):
        model.train()
        for batch_idx, (X, y) in enumerate(train_data_iter):
            
            ## use assign input to GDP or CPU
            inputs = X.to(device)#.float()
            labels = y.to(device)#.float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_outputs = loss_fn(outputs,labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
            ## print log info 
            if batch_idx+1 % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    (batch_idx+1) * len(inputs), len(train_data_iter.dataset),
                    100. * (batch_idx+1) / len(train_data_iter), np.mean(losses)*loss_scale)
                print(message)
                losses = []
        
        ## print training info after each epoch 
        total_loss /= (batch_idx + 1)
        message = 'Train eopch: {}/{}; \tLoss: {:.6f}'.format(i+1,n_epochs,total_loss*loss_scale)
        
        if do_eval:
            if i > warmup_n_epoch:
                _,_,train_acc = eval_model(model,train_data_iter)
                _,_,test_acc = eval_model(model,test_data_iter)
                message = 'Train eopch: {}/{}; \tLoss: {:.6f}; train accuarcy {:.4f}; test accuarcy {:.4f}'.format(i+1,
                                                                                                                n_epochs,
                                                                                                                total_loss*loss_scale,
                                                                                                                train_acc,
                                                                                                                test_acc)
        logger.info(message)  ## print out training message
        
        
        ## save model based on save criterion and warmup steps 
        if save_criterion == 'train_acc':
            if i > warmup_n_epoch:
                if best_save_criterion is None:
                    best_save_criterion = train_acc
                
                if train_acc>best_save_criterion:
                    best_save_criterion=train_acc
    
                    torch.save(model.state_dict(),save_model_path)
                    logger.info('best model saved in {}'.format(save_model_path))
                    
        elif save_criterion == 'test_acc':
            if i > warmup_n_epoch:
                if best_save_criterion is None:
                    best_save_criterion = test_acc
                
                if test_acc>best_save_criterion:
                    best_save_criterion=test_acc

                    torch.save(model.state_dict(),save_model_path)   
                    logger.info('best model saved in {}'.format(save_model_path))
                    
        elif save_criterion == 'train_loss':
            if i > warmup_n_epoch:
                if best_save_criterion is None:
                    best_save_criterion = total_loss
                
                if total_loss<best_save_criterion:
                    best_save_criterion=total_loss
 
                    torch.save(model.state_dict(),save_model_path)    
                    logger.info('best model saved in {}'.format(save_model_path))
                    
        
        total_loss=0
    
    return model, best_save_criterion

def eval_model(model,test_data_iter,device='cpu'):
    
    model.eval()
    ground_truth = []
    predicted_label = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(train_data_iter): 
            inputs = X.to(device)#.float()
            labels = y.to(device)#.float()
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            ground_truth.append(labels.tolist())
            predicted_label.append(predicted.tolist())
        
    label = np.array(ground_truth)
    lebel_hat = np.array(predicted_label)
    correct = (label == lebel_hat)
    acc = correct.sum()/correct.size
    
    return ground_truth,predicted_label,acc
            
            
#%%
if __name__ == '__main__':

    ## read training data 
    training_data_path = os.path.join(config.CRISIS_DATES,'train_data.pkl'.format())
    df = pd.read_pickle(training_data_path)
    dummies = df[['crisis_pre','crisis_tranqull','crisisdate']]
    df['label'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
    ## create label to description map 
    label2id, id2label = create_map(df,'label')
    df['label_num'] = df['label'].map(label2id)
    ## get train test data iter
    train_data_iter,test_data_iter = prepare_torch_training_data(df,'snip_emb','label_num',ratio=0.3)
    
    #%%
    
    input_size = 768
    hidden_size = [256,64]
    num_classes = 3
    learning_rate = 1e-3
    n_epochs = 100
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = Simple_nn_model(input_size,hidden_size,num_classes,dropout_p=0.1)
    model = Dynamic_simple_sequencial_model(input_size,hidden_size,num_classes,dropout_p=0.1).model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
    
    #%%
    
    model,_ = train_model(model,optimizer,criterion,n_epochs,train_data_iter,
                            do_eval=True, test_data_iter= test_data_iter,
                            save_criterion='train_acc',warmup_n_epoch=70,
                            save_model_path=os.path.join(config.TRAINED_DEEP_MODEL,
                                                         'simple_nn_model',
                                                         'weights_simple.pt'))
#    
#    #%%
#    _,_,acc = eval_model(model,test_data_iter)
#    print(acc)
