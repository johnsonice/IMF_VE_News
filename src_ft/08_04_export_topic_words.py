import gensim
import argparse
import config
import os 
import pandas as pd
from topic_model_utils import topic2df ##print_topics_gensim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corp_dir', default=os.path.join(config.BOW_TFIDF_DOCS,
                                                                                                'tfidf.mm'))
    parser.add_argument('-d', '--dictionary', action='store', dest='dict_dir',
                        default=os.path.join(config.BOW_TFIDF_DOCS, 'dictionary'))
    parser.add_argument('-clip', '--clip', action='store', dest='clip', default=None)
    parser.add_argument('-top', '--n_topics', action='store', nargs='+', dest='n_top_list', type=int,
                        default=[80, 100, 120, 140, 160, 180, 200])
    parser.add_argument('-p', '--passes', action='store', dest='passes', type=int, default=4)
    parser.add_argument('-m', '--model_folder', action='store', dest='model_folder', default=config.TOPIC_MODELS)
    parser.add_argument('-s', '--save', action='store', dest='save_dir', default=config.SEARCH_TERMS)
    parser.add_argument('-e', '--export_terms', action='store', dest='export_terms', default=True)
    args = parser.parse_args()
    
    
    ## get list of models to evaluate
    model_dirs = {}
    for n_top in args.n_top_list:
        model_name = 'lda_model_tfidf_{}_{}_{}'.format(n_top, args.clip, args.passes)
        model_dir=os.path.join(args.model_folder, model_name)
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
        print('\nexport search words for model : {}'.format(n_top))
        mod =  gensim.models.ldamodel.LdaModel.load(model_dirs[n_top])
        ## convert to df function 
        keys_df = topic2df(mod,total_topics=None,
                    weight_threshold=0.0001,
                    display_weights=False,
                    num_terms=30)
        
        keys_df.to_csv(os.path.join(args.save_dir,'model_{}_search_terms.csv'.format(n_top)),encoding='utf8',index=False)
        
    #%%
