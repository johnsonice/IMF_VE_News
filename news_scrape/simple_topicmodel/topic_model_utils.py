import os, sys,ssl,argparse
sys.path.insert(0,'../libs')
import itertools
from tqdm import tqdm
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import pandas as pd
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import json 

class Args:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                setattr(self, key, value)

def pack_update_param(param,coherence_scores,outlier_percent,n_topics,diversity_score):
    eval_dict = {
        'coherence': coherence_scores,
        'diversity': diversity_score,
        'outlier': outlier_percent,
        'number_topics': n_topics,
    }
    if param:
        param.update(eval_dict)
        return param
    else:
        return eval_dict

def hyper_param_permutation(hyper_params):
    ## prepare params for gride search ##
    assert isinstance(hyper_params,dict)
    param_names = list(hyper_params.keys())
    all_param_list = [hyper_params[k] for k in param_names]
    permu = list(itertools.product(*all_param_list))
    res = [dict(zip(param_names,i)) for i in permu]
    return res

def prepare_docs_for_coherence_eval(docs,topics,probabilities,model):
    documents = pd.DataFrame({"Document": docs,
                          "ID": range(len(docs)),
                          "Topic": topics,
                          "Topic_prob": probabilities})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    #print(documents_per_topic.head())
    # Extract vectorizer and analyzer from BERTopic
    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)
    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic) if words!=''] 
                for topic in range(len(set(topics))-1)]
    topic_words = [t for t in topic_words if len(t) >0] ## for some reason some topics has all "" as topic words

    return topic_words,tokens,corpus,dictionary

def get_coherence_score(topic_words,tokens,corpus,dictionary,n_workers=-1):
    # Evaluate
    # print("n_workers: {}".format(n_workers))
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v',
                                    processes=n_workers)
    coherence = coherence_model.get_coherence()
    
    return coherence

def eval_coherence_score(docs,topics,probabilities,model,n_workers=-1):
    topic_words,tokens,corpus,dictionary = prepare_docs_for_coherence_eval(docs,topics,probabilities,model)
    coherence = get_coherence_score(topic_words,tokens,corpus,dictionary,n_workers=n_workers)
    
    return coherence
    

def BERTopic2OCTIS_output(topic_model):
    topics_rep = topic_model.get_topics()
    OCTIS_topics = [[k[0] for k in top] for top in list(topics_rep.values())]
    return {'topics':OCTIS_topics}

def eval_diversity_score(topic_model):
    
    diversity_metric = TopicDiversity(topk=topic_model.top_n_words)
    octis_model_output = BERTopic2OCTIS_output(topic_model)
    topic_diversity_score = diversity_metric.score(octis_model_output) # Compute score of the metric

    return topic_diversity_score

def model_setup(train_args):
    '''
    initialize topic model with training args 
    '''
    ## Step 3 - set up umap for reduction 
    ## if you want to make it reproducable; set random state in umap to be fixed 
    umap_model = UMAP(n_neighbors=train_args.n_neighbors,   # local neighborhood size for UMAP. default is 15, larget mean more global structure
                                                            # This is the parameter that controls the local versus global structure in data
                    n_components=train_args.n_components,   # output dimension for UMAP
                    min_dist=0,             # to allow UMAP to place points closer together (the default value is 1.0)
                    metric='cosine',        # use cosine distance 
                    random_state=42)        # fix random seed 

    ## Step 3 - Cluster reduced embeddings
    ## see link for more param selection:
    ## https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#parameter-selection-for-hdbscan
    hdbscan_model = HDBSCAN(min_cluster_size=train_args.min_cluster_size,  #the minimum number of documents in each cluster, for larger data, this should be larger
                            min_samples=train_args.min_samples,            #controls the number of outliers. It defaults to the same value as min_cluster_size. 
                                                            #The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, 
                                                            #and clusters will be restricted to progressively more dense areas
                                                            #we should keep this constant when tuning other parameters 
                            metric=train_args.metric,       #I guess we can try cosine here ? 
                            cluster_selection_method='eom', #The default method is 'eom' for Excess of Mass, the algorithm described in How HDBSCAN Works https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html.
                            prediction_data=True)

    representation_model = KeyBERTInspired()

    ## Step 4-5 - prepare param for c-tfidf for topic representation 
    ## additional topic merging will be done by compare distance (based on bog method on c-tfidf), set to auto will use HDBSCAN
    ## remove stop words when fingding keys for topic representation ; sbert will still use full sentence 
    vectorizer_model = CountVectorizer(ngram_range=(1, 2),
                                        stop_words="english",       # you can also provide a customized list 
                                        min_df=train_args.min_df,                  # set min number of word frequency
                                        #vacabulary=custom_vocab,   # you can also use a customized vocabulary list, 
                                                                    # e.g use keybert: https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html#keybert-bertopic
                                        )               
    ## ctfidf param can be pass in topicbert main function 


    if train_args.verbose:
        print('use {} as embeding model'.format(train_args.model_checkpoint))
    emb_model = SentenceTransformer(train_args.model_checkpoint)  

    ## call main function 
    topic_model = BERTopic(
                    umap_model=umap_model,              # Reduce dimensionality 
                    hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
                    vectorizer_model=vectorizer_model,  # Step 4,5 - use bang of words and ctfidf for topic representation
                    embedding_model=emb_model,
                    representation_model=representation_model,
                    #diversity= train_args.diversity,            in 0.14, removed # Step 6 - Diversify topic words ; maybe also try 0.5?
                    ## other params 
                    language="English",
                    verbose=train_args.verbose,
                    top_n_words=train_args.top_n_words,         # number of topic words to return; can be changed after model is trained 
                                                                # https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.update_topics
                    min_topic_size=train_args.min_cluster_size, # this should be the same as min_cluster_size in HDBSCAN
                    nr_topics=train_args.nr_topics,               # number of topics you want to reduce to ; auto will use results from HDBSCAN on c-tfidf
                    calculate_probabilities = train_args.calculate_probabilities, # Whether to calculate the probabilities of all topics per document instead of the probability of the assigned topic per document. 
                    )
    
    return topic_model 

def train_topic_model(args,docs,embeddings):
    topic_model = model_setup(args)
    topics, probabilities = topic_model.fit_transform(docs,embeddings)
    if isinstance(probabilities,np.ndarray):
        probabilities = probabilities.tolist()

    return topics,probabilities,topic_model

def eval_topic_model(docs,topics,probabilities,topic_model,n_workers=1):
    coherence_scores = eval_coherence_score(docs,topics,probabilities,topic_model,n_workers=n_workers)
    topic_freq = topic_model.get_topic_freq()
    outlier_percent = topic_freq['Count'][topic_freq['Topic'] == -1].iloc[0]/topic_freq['Count'].sum()
    n_topics = len(topic_model.get_topic_freq())
    diversity_score = eval_diversity_score(topic_model)
    
    return coherence_scores,outlier_percent,n_topics,diversity_score

def train_and_eval(args,docs,embeddings,n_workers=1):
    try:
        topics,probabilities,topic_model = train_topic_model(args,docs,embeddings)
    except Exception as e:
        print('--Topic Model training error -- : {}'.format(e))
        topics,probabilities,topic_model = (None,None,None)
    
    if topic_model is not None:
        coherence_scores,outlier_percent,n_topics,diversity_score = eval_topic_model(docs,topics,probabilities,
                                                                                topic_model,n_workers=n_workers)
    else:
        coherence_scores,outlier_percent,n_topics,diversity_score = (None,None,None,None)