#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:16:35 2019

@author: chuang
"""

# -*- coding: utf-8 -*-

import pickle
import os 
import pandas as pd
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
from stream import MetaStreamer_fast as MetaStreamer
import ujson as json
import config
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
sns.set(color_codes=True)

#%%
def get_word_vecs(vecs,keyword,topn=20):
    try:
        sim_list = [s[0] for s in vecs.wv.most_similar(keyword, topn=topn)]
    except:
        sim_list = [s[0] for s in vecs.wv.most_similar(keyword.split("_"), topn=topn)]

    #print(sim_list)
    sim_list.append(keyword)
    
    sim_dict = {k:vecs.wv.get_vector(k) for k in sim_list}
    
    return sim_dict


#%%
def prepare_chart_data(vecs,keys=['fear'],topn=15,method='tsne'):
    words_dict=dict()
    for key in keys:
        temp_dict = get_word_vecs(vecs,key,topn=topn)
        for wd,vs in temp_dict.items():
            words_dict[wd] = vs
    
    words_keys = list(words_dict.keys())
    #print(words_keys)
    
    vectors = np.vstack([words_dict[k] for k in words_keys])    
    if method.lower() == 'pca':
        print('reduce dimention using pca')
        model = PCA(n_components=2,random_state=0)
    else:
        print('reduce dimention using tsne')
        model = TSNE(n_components=2,random_state=0)
    vectors_transformed = model.fit_transform(vectors)
    x = vectors_transformed[:,0] 
    y = vectors_transformed[:,1] 
    return words_keys,x,y

#%%
#def make_chart_with_ax(ax,x,y,group):
#    sns.kdeplot(x,y,cmap="Blues",shade=True,shade_lowest=True,ax = ax)
#    sns.scatterplot(x,y,hue=group,palette='Set2', legend = False,ax = ax)
#    sns.scatterplot(x,y,legend = False,ax = ax)
#    return ax

def make_chart_with_ax_2(ax,chart_data):
    sns.kdeplot(chart_data['x'],chart_data['y'],cmap="Blues",shade=True,shade_lowest=True,ax = ax)
    #sns.scatterplot(x,y,hue=group,palette='Set2', legend = False,ax = ax)
    sns.scatterplot(x = 'x',y = 'y',hue='group',palette=sns.color_palette('hls',n_colors=3),
                    legend = False,s=70,ax = ax,data=chart_data)
    return ax
#%%
if __name__ == "__main__":
    
    vecs = KeyedVectors.load(config.W2V)
    #%%
    terms = ['fear','worry','concern','risk','threat','warn','maybe','may','possibly','could',
             'perhaps','uncertain','say','feel','predict','tell','believe','think','recession',
             'financial_crisis','crisis','depression','shock']
    #for t in terms:
    words_keys,x,y = prepare_chart_data(vecs,keys=['crisis','concern','predict'],topn=10,method='pca')
    ## add grouping variable
    group = [1]*11 + [2]*11+[3]*11
    clean_data = zip(words_keys,x,y,group)
    clean_data = [(k,i,v,g) for k,i,v,g in clean_data if k not in ['cri_sis','wonder_whether','cnsis','wamed','oncern']]
    print(words_keys)
    #words_keys,x,y,g  = map(list,zip(*clean_data))
    chart_data = pd.DataFrame(clean_data,columns=['keys','x','y','group'])
#%%
    fig,ax = plt.subplots(figsize=(16,10))
    ax = make_chart_with_ax_2(ax,chart_data)
#    for k,i,j in zip(words_keys,x,y):
#        ax.text(i+0.01,j+0.01,k)
    ax.set(xlabel='PCA 1',ylabel='PCA 2')
    fig.savefig('figs/crisis_concern_predict_no_text.png')