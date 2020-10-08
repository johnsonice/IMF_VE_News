import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import pandas as pd
import config
from gensim.models.keyedvectors import KeyedVectors


def get_sim_words(vecs,wg,topn):
    if not isinstance(wg,list):
        wg = [wg]
    try:
        sims = [w[0] for w in vecs.wv.most_similar(wg, topn=topn)]
    except KeyError:
        try:
            wg_update = list()
            for w in wg:
                wg_update.extend(w.split('_'))
            sims = [w[0] for w in vecs.wv.most_similar(wg_update, topn=topn)]
            print('Warning: {} not in the vocabulary, split the word with _'.format(wg))
        except:
            print('Not in vocabulary: {}'.format(wg_update))
            return wg
    words = sims + wg
    return words


vecs = KeyedVectors.load(config.W2V)

targets = config.targets()
target_sim_df = pd.DataFrame(index=range(15), columns=targets)

for target in targets:
    target_sim_df[target] = get_sim_words(vecs, target, 15)

target_sim_df.to_csv('/home/apsurek/IMF_VE_News/research/w2v_compare/targets_to_sims_mapping.csv')