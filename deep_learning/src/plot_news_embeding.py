#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 12:32:38 2020

@author: chuang
Plot news embeding
"""
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import config
import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)

## plot 

def prepare_chart_data(vectors,method='tsne'):
    
    if method.lower() == 'pca':
        print('reduce dimention using pca')
        model = PCA(n_components=2,random_state=0)
    else:
        print('reduce dimention using tsne')
        model = TSNE(n_components=2,random_state=0)
    vectors_transformed = model.fit_transform(vectors)
    x = vectors_transformed[:,0] 
    y = vectors_transformed[:,1] 
    return x,y

def make_chart_with_ax(ax,chart_data,x,y,color_group):
    #sns.kdeplot(chart_data['x'],chart_data['y'],cmap="Blues",shade=True,shade_lowest=True,ax = ax)
    #sns.scatterplot(x,y,hue=group,palette='Set2', legend = False,ax = ax)
    #n_colors = len(chart_data[color_group].unique())
    sns.scatterplot(x = x,y = y,hue=color_group,
                    s=70,ax = ax,data=chart_data) 
    #legend = False,palette=sns.color_palette('hls',n_colors=n_colors),
    return ax
#%%
    
if __name__ == '__main__':
    
    training_data_path = os.path.join(config.CRISIS_DATES,'train_data.pkl'.format())
    fig_path = config.CRISIS_DATES

    df = pd.read_pickle(training_data_path)

    title_vectors = np.array(df.title_emb.tolist())
    snip_vectors= np.array(df.snip_emb.tolist())

    title_x, title_y = prepare_chart_data(title_vectors,'tsne')

    snip_x, snip_y = prepare_chart_data(snip_vectors,'tsne')

    df['title_x'],df['title_y'],df['snip_x'],df['snip_y'] = [title_x, title_y,snip_x, snip_y ]

    ## create color group
    dummies = df[['crisis_pre','crisis_tranqull','crisisdate']]
    df['color_group'] = pd.Series(dummies.columns[np.where(dummies!=0)[1]])
    
    #%%
    
    ## plot overall snip embemding
    fig,ax = plt.subplots(figsize=(16,10))
    ax = make_chart_with_ax(ax,df,'snip_x','snip_y','color_group')
    plt.title('Overall Snip Embeding Plot')
    fig.savefig(os.path.join(fig_path,'fig','overall_snip_embed_plot.png'))

    #by country plot 
    for c in df.country.unique():
        country_df = df[df.country==c]
        fig,ax = plt.subplots(figsize=(16,10))
        ax = make_chart_with_ax(ax,country_df,'snip_x','snip_y','color_group')
        plt.title('{} Snip Embeding Plot'.format(c))
        fig.savefig(os.path.join(fig_path,'fig','{}_snip_embed_plot.png'.format(c)))

#%%
    
#    ## plot overall title embemding
#    fig,ax = plt.subplots(figsize=(16,10))
#    ax = make_chart_with_ax(ax,df,'title_x','title_y','color_group')
#    plt.title('Overall title Embeding Plot')
#    fig.savefig(os.path.join(fig_path,'fig','overall_title_embed_plot.png'))
#
#    #by country plot 
#    for c in df.country.unique():
#        country_df = df[df.country==c]
#        fig,ax = plt.subplots(figsize=(16,10))
#        ax = make_chart_with_ax(ax,country_df,'title_x','title_y','color_group')
#        plt.title('{} title Embeding Plot'.format(c))
#        fig.savefig(os.path.join(fig_path,'fig','{}_title_embed_plot.png'.format(c)))
    
    #%%
#    for k,i,j in zip(words_keys,x,y):
#        ax.text(i+0.01,j+0.01,k)
#    ax.set(xlabel='TSNE 1',ylabel='TSNE 2')
    #fig.savefig('figs/crisis_concern_predict_no_text.png')