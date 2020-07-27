#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 12:04:21 2020

@author: chuang
"""

## train simple model 
import torch 
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import config
import os
import numpy as np
from sklearn.model_selection import train_test_split

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
    

class Simple_nn_model(nn.Module):
    
    def __init__(self,input_size,hidden_size,num_classes,dropout_p=0.1,batchnorm=True,layernorm=False):
        super(Simple_nn_model,self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1=nn.Linear(input_size,hidden_size[0])
        self.fc2=nn.Linear(hidden_size[0],hidden_size[1])
        self.fc3=nn.Linear(hidden_size[1],num_classes)
        #self.fc4=nn.Linear(hidden_size[2],num_classes)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(num_features = hidden_size[0])
            self.bn2 = nn.BatchNorm1d(num_features = hidden_size[1])
            #self.bn3 = nn.BatchNorm1d(num_features = hidden_size[2])
        else:
            self.bn1,self.bn2,self.bn3 = None,None,None
            
    def forward(self,x):
        out = x 
        #out = self.dropout(out)
        
        if not self.bn1 is None:
            out=F.relu(self.bn1(self.fc1(out)))
        else:
            out=F.relu(self.fc1(out))
        out = self.dropout(out)
        
        if not self.bn2 is None:    
            out = F.relu(self.bn2(self.fc2(out)))
        else:
            out=F.relu(self.fc2(out))
        #out = self.dropout(out)
        
#        if not self.bn3 is None:    
#            out = F.relu(self.bn3(self.fc3(out)))
#        else:
#            out=F.relu(self.fc3(out))
 
        out = self.fc3(out)
        return out 

def train_model(model,optimizer,loss_fn,n_epochs,train_data_iter,
                log_interval=100,loss_scale=1,device="cpu",
                do_eval=False,test_data_iter=None):
        
    model.train()
    losses = []
    total_loss = 0
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
        message = 'Train eopch: {}/{}; \tLoss: {:.6f}'.format(i,n_epochs,total_loss*loss_scale)
        if do_eval:
            _,_,train_acc = eval_model(model,train_data_iter)
            _,_,test_acc = eval_model(model,test_data_iter)
            message = 'Train eopch: {}/{}; \tLoss: {:.6f}; train accuarcy {:.4f}; test accuarcy {:.4f}'.format(i,
                                                                                                            n_epochs,
                                                                                                            total_loss*loss_scale,
                                                                                                            train_acc,
                                                                                                            test_acc)
        
        print(message)
        total_loss=0
    
    return model 

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
    model = Simple_nn_model(input_size,hidden_size,num_classes,dropout_p=0.1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate) 
    
    #%%
    
    model = train_model(model,optimizer,criterion,n_epochs,train_data_iter,
                        do_eval=True, test_data_iter= test_data_iter)
#    
#    #%%
#    _,_,acc = eval_model(model,test_data_iter)
#    print(acc)
