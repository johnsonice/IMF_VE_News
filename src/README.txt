Author: Sean Simpson
Date: Aug 3rd, 2018

# ===========
# About
# ===========
This directory contains scripts written as part of an effort to detect natural 
language indicators of financial crisis. An overview and brief explanation of 
each of these scripts can be found below. A more thorough description of each 
script and the functions contained within can be found in the docstrings 
throughout the scripts themselves. 

If a script is present in the dir but not listed here, it is either because 
it was created after this readme was written, or because it is obsolete or 
unused in the main analysis. 

Note: none of the plotting scripts will work on the AWS EC2 instance unless you
are properly set up for x11 forwarding. 


#---------------
# corpus scripts
#---------------
The following is a list of scripts that manipulate or parse the corpus in various ways.

-- avro2json.py:
    converts avro files (original format in which newspaper corpus was given to IMF) into
    json files (1 file per article)

-- stream.py: 
    provides several types of streamer classes. Used to stream data from corpus in a memory friendly way. 
    Also provides many options for filtering those docs streamed.

-- corpus_doc_details: 
    Creates a DF containing details for each document in the corpus for quick reference downstream.  the 
    returned dataframe contains a string rep. of the month, quarter, and week of publication, doc_id number, 
    and whether or not it is a crisis doc for any countries.

-- corpus_phrases.py: 
    used to train ngram language models on corpus for phrase detection

-- corpus_preprocessing.py: 
    provides some (extremely) light preprocssing and text normalization.

-- corpus_split_train_test.py: 
    symlinks a random 20% of the corpus docs into a 'test' dir and symlinks the other 80% into a 'train' dir. 
    You may specify the percentage of your corpus you want to use for testing-- 20% is the default. 

-- corpus_tfidf.py: 
    takes a supplied corpus and creates:
     - a dictionary 
     - a BOW formatted version of the corpus
     - a TFIDF transformed version of the BOW corpus object. 

-- crisis_points.py: 
    provides a dictionary of all the start and peak dates for the crises listed in K&R1999


# -----------------------------------------
# scripts used in semantic cluster analysis
# -----------------------------------------
-- frequency_country_specific_freqs.py: 
    Use this to extract a frequency time series for each word in corpus based just on those docs which pertain 
    to a specific country. The country specific word frequency data created here is used in most other freq scripts.

-- evaluate.py: 
    provides a useful collection of functions used to calculate recall, precision, and fscore of aggregate 
    freq spikes for a given group of words.

-- frequency_eval.py: 
    evaluate semantic term clusters seeded by supplied terms

-- frequency_eval_expert-terms.py: 
    evaluate semantic term clusters seeded by terms on supplied term list

-- frequency_most_similar.py: 
    plot aggregate freq of semantic cluster seeded by supplied term as well as 
    crisis points for supplied countries and significant increases in frequency.

-- frequency_utils: 
    collection of useful functions used in frequency scripts

-- vectorize_words.py:
    used to train vector space models for later similarity querries. 


# -----------------------------------------
# scripts used in topic cluster analysis
# -----------------------------------------
-- topic_lda_train.py: 
    Use this to train topic models

-- topic_lda_coherence.py: 
    use this to compare trained models wrt topic coherence

-- topic_lda_analytics.py: 
    Use this to get country specific and aggregate freq analysis for each topic in topic model

-- topic_lda_summarize.py: 
    use this to evaluate aggregate recall, precision, and f2 over all topics within a given model 
    (must be run after topic_lda_analytics.py)

-- topic_lda_print_topics.py: 
    this is more of a convenience script. Use it to print out the words belonging to each topic 
    in a given topic model.

















