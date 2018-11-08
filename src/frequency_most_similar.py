"""
Examine frequency of semantically related words in the corpus
"""
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
import pandas as pd
#from frequency_utils import plot_frequency
from plot_utils import crisis_plot,plot_frequency

#%%
def plot_similar_freqs(target_words, vecs, country, country_freqs, topn=10, roll_avg=True, roll_window=20):
    assert type(target_words) in (list, str)
    if vecs is not None:
        assert type(vecs) == Word2Vec

    # Find topn most similar words according to VSM
    target_words = [target_words] if isinstance(target_words, str) else target_words
#    words = [w[0] for w in vecs.most_similar(target_words, topn=topn)]
#    words += target

    if vecs is not None:
        try:
            words = [w[0] for w in vecs.most_similar(target_words, topn=topn)]
            words += target
        except:
            words = target_words
    else:
        words = target_words
    
    print(words,roll_avg)
    # Generate plot
    if not roll_avg:
        fig = plot_frequency(country_freqs, words=words, country=COUNTRY)
    else:
        word_freqs = [country_freqs.loc[word] for word in words if word in country_freqs.index and sum(country_freqs.loc[word] != 0)]
        grp_freq = sum(word_freqs)
        grp_rolling = grp_freq.rolling(window=roll_window).mean()
        fig = crisis_plot(grp_rolling, country=COUNTRY,roll_avg=False)

    # return plot
    fig.suptitle("{} Frequency of top {} words most similar to {}".format(country, topn, target))
    return fig


if __name__ == '__main__':
    #MODEL = "/home/ubuntu/Documents/v_e/models/vsms/word_vecs_5_10_200"
    MODEL = None
    COUNTRY = 'argentina'
    if MODEL is not None:
        vecs = KeyedVectors.load(MODEL)
    else:
        vecs = None
        
    country_freqs = pd.read_pickle("../data/frequency/{}_processed_json_month_word_freqs.pkl".format(COUNTRY))

    target = 'citizen imf summit east'.split(" ")
    plot_similar_freqs(target, vecs=vecs, country=COUNTRY, country_freqs=country_freqs,
                       roll_avg=False, roll_window=5)
