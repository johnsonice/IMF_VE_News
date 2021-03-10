import numpy
import pandas
import pickle

# Configure speed test inputs




# Test NLTK speed
import nltk
nltk.download('vader_lexicon') # If lexicon not downloaded
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Use sentence streamer
sid.polarity_scores(sentence)






import textblob
import flair
import affin

