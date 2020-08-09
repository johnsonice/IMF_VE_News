#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:19:26 2020

@author: chuang
"""

## training utils
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(model,optimizer,loss_fn,n_epochs,train_data_iter,
                log_interval=100,loss_scale=1,device="cpu",
                do_eval=False,test_data_iter=None,
                save_criterion=None,warmup_n_epoch=10,
                save_model_path=None, save_mode = 'full'):
        
    model.train()
    losses = []
    total_loss = 0
    best_save_criterion = None
    
    return_metric = []
    
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
            if i >= warmup_n_epoch:
                _,_,train_acc,train_loss = eval_model(model,train_data_iter,loss_fn)
                _,_,test_acc,test_loss = eval_model(model,test_data_iter,loss_fn)
                return_metric.append((i,train_loss,test_loss))
                message = 'Train eopch: {}/{}; \tLoss: {:.6f}; train accuarcy {:.4f};test loss: {:.4f}; test accuarcy {:.4f}'.format(i+1,
                                                                                                                n_epochs,
                                                                                                                train_loss*loss_scale,
                                                                                                                train_acc,
                                                                                                                test_loss,
                                                                                                                test_acc)
        logger.info(message)  ## print out training message
        
        
        ## save model based on save criterion and warmup steps 
        if save_criterion == 'train_acc':
            if i >= warmup_n_epoch:
                if best_save_criterion is None:
                    best_save_criterion = train_acc
                
                if train_acc>best_save_criterion:
                    best_save_criterion=train_acc
                    save_model(model,save_model_path,save_mode=save_mode)
                    #torch.save(model.state_dict(),save_model_path)
                    logger.info('best model saved in {}'.format(save_model_path))
                    
        elif save_criterion == 'test_acc':
            if i >= warmup_n_epoch:
                if best_save_criterion is None:
                    best_save_criterion = test_acc
                
                if test_acc>best_save_criterion:
                    best_save_criterion=test_acc
                    save_model(model,save_model_path,save_mode=save_mode)
                    #torch.save(model.state_dict(),save_model_path)   
                    logger.info('best model saved in {}'.format(save_model_path))
                    
        elif save_criterion == 'train_loss':
            if i >= warmup_n_epoch:
                if best_save_criterion is None:
                    best_save_criterion = total_loss
                
                if total_loss<best_save_criterion:
                    best_save_criterion=total_loss
                    save_model(model,save_model_path,save_mode=save_mode)
                    #torch.save(model.state_dict(),save_model_path)    
                    logger.info('best model saved in {}'.format(save_model_path))
                    
        
        total_loss=0
    if do_eval:
        return_metric_df = pd.DataFrame(return_metric,columns=['batch_num','train_loss','test_loss'])
        return model, best_save_criterion,return_metric_df
    else:
        return model, best_save_criterion


def save_model(model,save_model_path,save_mode='full'):
    '''
    save model mode  = [full, state_dict, checkpoint] ## for convience, we default to full 
    check https://pytorch.org/tutorials/beginner/saving_loading_models.html?hilight=load
    '''
    if save_mode == 'full':
        torch.save(model,save_model_path)
    elif save_mode == 'state_dict':
        torch.save(model.state_dict(),save_model_path) 
    elif save_mode == 'checkpoint':
        raise Exception('not tyet implemented')
    
    return None
        

def eval_model(model,test_data_iter,loss_fn=None,device='cpu'):
    
    model.eval()
    ground_truth = []
    predicted_label = []
    losses = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_data_iter): 
            inputs = X.to(device)#.float()
            labels = y.to(device)#.float()
            
            outputs = model(inputs)
            loss_outputs = loss_fn(outputs,labels)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            ground_truth.append(labels.tolist())
            predicted_label.append(predicted.tolist())
    
    average_loss = np.mean(losses)
    label = np.array(ground_truth)
    lebel_hat = np.array(predicted_label)
    correct = (label == lebel_hat)
    acc = correct.sum()/correct.size
    
    return ground_truth,predicted_label,acc,average_loss