import gensim
import argparse
from stream import DocStreamer_fast
import pandas as pd


def eval(model, texts, coherence_type):
    coh_model = gensim.models.coherencemodel.CoherenceModel(model=model, corpus=texts, 
                                                            coherence=coherence_type)
    return coh_model.get_coherence()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_list', action='store', nargs='+', dest='mlist')
    parser.add_argument('-d', '--dictionary', action='store', dest='corp_dict', help='Gensim Dictionary object path',
                       default='../data/corpus_dict_test')
    parser.add_argument('-t', '--texts', action='store', dest='texts', help='streamer to test on',
                       default='../test_docs')
    parser.add_argument('-c', '--coherence', action='store', dest='coherence', help='(str) coherence type to eval',
                       default='c_v')
    opts = parser.parse_args()

    # Setup
    phraser = '/home/ubuntu/Documents/v_e/models/ngrams/2grams_default_10_20_NOSTOP'
    stream = DocStreamer_fast('../test_docs', language='en', phraser=phraser, verbose=True, lemmatize=False) 

    # Eval
    coherence = {}
    mlist = opts.mlist
    for mod_name in mlist:
        mod = gensim.models.ldamodel.LdaModel.load(mod_name)
        coh_model = gensim.models.coherencemodel.CoherenceModel(model=mod, texts=stream, coherence='c_v')
        coherence[mod_name] = coh_model.get_coherence()
    coherence = pd.Series(coherence)
    print(coherence)
    coherence.to_csv('../eval/coherence.csv')
    


