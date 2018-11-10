from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import numpy as np
import time

# ---- Constants
analyzer = SentimentIntensityAnalyzer()


def doc_sentiment(text, sent_type='compound'):
    try:
        sents = sent_tokenize(text)
        doc_sent = 0
        for sent in sents:
            scores = analyzer.polarity_scores(sent)
            doc_sent += scores[sent_type]
        doc_sent = doc_sent / len(sents)
        return doc_sent
    except:
        return np.nan


if __name__ == '__main__':
    import pandas as pd
    from stream import FileStreamer

    CORPUS = '/home/ubuntu/Documents/v_e/cleaned'
    TIMES = '/home/ubuntu/Documents/v_e/data/frequency/corpus_time_series.pkl'

    sents = pd.read_pickle(TIMES)
    files = FileStreamer(CORPUS)

    print("calculating sentiment...")
    total = 0
    ids_vals = []
    for doc in files:
        print("\r{} docs of {} done".format(total, files.flist_length), end='')
        id = doc['an']
        text = doc['body']
        if text:
            ids_vals.append((id, doc_sentiment(text)))
        total += 1
    sent_frame = pd.DataFrame(ids_vals, columns=['id','sentiment'])
    sent_frame.set_index('id', inplace=True)
    sents = pd.concat([sents, sent_frame], axis=1)

    out_file = '/home/ubuntu/Documents/v_e/data/doc_sentiment'
    sents.to_pickle(out_file + '.pkl')
    sents.to_csv(out_file + '.csv')

