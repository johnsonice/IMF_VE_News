"""
Calculate monthly sentiment in corpus using docs relevant to taget country
"""
import pandas as pd
from matplotlib import pyplot as plt
from stream import FileStreamer
from crisis_points import crisis_points

# Data
sentiment_data = "/home/ubuntu/Documents/v_e/data/doc_sentiment.pkl"
sent = pd.read_pickle(sentiment_data)

# Params
COUNTRY = 'argentina'

# Identify relevant docs, pull out sentiment from each
if COUNTRY:
    CORPUS = '/home/ubuntu/Documents/v_e/cleaned'
    streamer = FileStreamer(CORPUS, language='en', regions=['arg'], region_inclusive=True,
                            title_filter=['argentina', 'argentine', 'argentinian'])
    f_index = []
    for f in streamer:
        id = f['an']
        f_index.append(sent.index.get_loc(id))
    sent = sent.iloc[f_index]

# Save sentiment dataframe
monthly_sent = sent.groupby('month')['sentiment'].mean()
outf = "/home/ubuntu/Documents/v_e/data/sentiment/{}_sentiment.pkl".format(COUNTRY)
monthly_sent.to_pickle(outf)

# Plot monthly sentiment
plt.plot(monthly_sent.values)
plt_indices = [i for i in range(len(monthly_sent)) if i % 2 == 0]
plt_labs = list(monthly_sent.index[plt_indices])
plt.xticks(plt_indices, plt_labs, rotation=90, fontsize=6)
plt.show()