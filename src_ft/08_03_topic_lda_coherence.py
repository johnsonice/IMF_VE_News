import gensim
import argparse
import config
import os 
import pandas as pd
from topic_model_utils import topic2df ##print_topics_gensim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corp_dir', default=os.path.join(config.BOW_TFIDF_DOCS,'tfidf.mm'))
    parser.add_argument('-d', '--dictionary', action='store', dest='dict_dir', default=os.path.join(config.BOW_TFIDF_DOCS,'dictionary'))
    parser.add_argument('-clip', '--clip', action='store', dest='clip', default=None)
    parser.add_argument('-top', '--n_topics', action='store', nargs='+', dest='n_top_list', type=int, default=[80,100,120,140,160,180,200])
    parser.add_argument('-p', '--passes', action='store', dest='passes', type=int, default=4)
    parser.add_argument('-s', '--save', action='store', dest='save_dir', default=config.TOPIC_MODELS)
    args = parser.parse_args()
    
    
    ## get list of models to evaluate
    model_dirs = {}
    for n_top in args.n_top_list:
        model_name = 'lda_model_tfidf_{}_{}_{}'.format(n_top, args.clip, args.passes)
        model_dir=os.path.join(args.save_dir, model_name)
        model_dirs[n_top] =model_dir
    print(model_dirs)
    
    # load gensim corpus 

    #corp = gensim.corpora.MmCorpus(args.corp_dir)
    corp = gensim.corpora.MmCorpus(os.path.join(config.BOW_TFIDF_DOCS,'bow.mm'))
    dictionary = gensim.corpora.Dictionary.load(args.dict_dir)
    #%%
    #############################
    ## export topic keys to csv##
    #############################
    for n_top,model_dir in model_dirs.items():
        print('\nexport keywords for model : {}'.format(n_top))
        mod =  gensim.models.ldamodel.LdaModel.load(model_dirs[n_top])
        ## convert to df function 
        keys_df = topic2df(mod,total_topics=None,
                    weight_threshold=0.0001,
                    display_weights=True,
                    num_terms=30)
        
        keys_df.to_csv(os.path.join(args.save_dir,'coherence_eval/model_{}.csv'.format(n_top)),encoding='utf8')
        
    #%%
    ###############################
    #### evaluate coherence score##
    ###############################
    coherence = {}
    # load model  
    for n_top,model_dir in model_dirs.items():
        print('Evaluate model: n_topic = {}'.format(n_top))
        mod = gensim.models.ldamodel.LdaModel.load(model_dir)
        # evaluate model
        coh_model = gensim.models.coherencemodel.CoherenceModel(model=mod, corpus=corp, coherence='u_mass',processes=os.cpu_count()-1)
        score = coh_model.get_coherence()
        coherence[n_top] = score
    
    df = pd.Series(coherence)
    print(df)
    df.plot()
    df.to_csv(os.path.join(args.save_dir,'coherence_eval/coherence.csv'),encoding='utf8')

    #%%
    #########################
    # export lda vis #######
    ########################
    import pyLDAvis
    import pyLDAvis.gensim
    
    for n_top,model_dir in model_dirs.items():
        print('\nexport lda viz for model : {}'.format(n_top))
        mod =  gensim.models.ldamodel.LdaModel.load(model_dirs[n_top])
        ## create viz_data
        viz_data = pyLDAvis.gensim.prepare(mod,corp,dictionary,sort_topics=True)
        print('prepare data done')
        pyLDAvis.save_html(viz_data,os.path.join(args.save_dir,"ldaviz_t{}.html".format(n_top)))
        print('data saved')