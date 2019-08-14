# News Analyais Vulnerbility Exercise

[Basic descriptions]
.....
....
### Environment Set up and some bash scripts 
We are working on linux environment, it has not been tested in a windows environment. 
Installed packeages needed for this project, assuming anaconda is installed in your environment

```sh
$ conda install -c conda-forge spacy langdetect
$ conda install gensim nltk numpy pandas jupyter notebook spyder glob2 scikit-learn seaborn lxml beautifulsoup
    ## install spacy dependecy 
$ python -m spacy download en_core_web_lg
    ## if your network is behind a proxy, this may not work. you will have to change your default pip config to have proxy information and trusted-host information. 
    ## or you can download the model file and install locally, for instance you have downlaod the en_core_web_sm-2.0.0.tar.gz file 
$ pip install /Users/you/en_core_web_sm-2.0.0.tar.gz
    ## set up shortcut link to load installed package as "en_default"
$ python -m spacy link en_core_web_lg en_core_web_lg
```

### File Structures

* [./scripts] - contains all scripts needs to transform raw data into a universal json form for processing. It also inlude a simple zip and unzip bash file in case your raw data come in zipped files.
* [./src_ft] - contains all pythons files you need to replicate paper results
    - [config.py] - Set up all your parameters choices. check all needed folders and create if not exist
    - [01_corpus_preprocessing.py] - very basic raw text clean up and lemmentization, save all raw data in a processed folder. 
    - [02_0_corpus_doc_details.py] - go through all documents and extract time and other useful metadata and save them to a pandas dataframe
    - [02_1_meta_summary.py] - this is an augumentation step on top of original metadata extraction. it goes through all documents again, get country information. and created a time series of document number. 
    - [03_corpus_phrases.py] - trains a phrase model based on our corpus, and save weights in model filder as input for document streamer. if you have large number of documents, so large that it doesn't fit into your memory, the function will automaticall process your files by chunks and do an online training of the phrase model.
    - [04_1_vectorize_words.py] - trains a w2v model based on our corpus. same model in model folder. it also does online training if input is too large. 
    - [05_corpus_tfidf.py] - transform all raw data into tfidf representation for each document
    - [06_frequency_country_specific_freqs.py] - create an aggregated bow representation by country by period
    - [06_1_export_keywords_timeseries.py] - given targets (keywords) and countries. it will create times series data keywords frequencies
    - [07_01_frequency_eval.py] - based on crisis events happend in the past, evaluate the keywords we picked up. See if they actually have any predictive power on crisis events. 

[...to be continued...]

### Todos
1.	Try shorter window size when running evaluation ( 1y, 6 month, 3 month), see if results are robust against benchmark.
2.	Try use monthly data and repeat the same exercise. 
3.	Do a weighted sum of word frequencies based on cosine distance from seed words. Try aggregate each concepts by using weighted average. 
4.	Try other seed words (maybe words for positive/over-optimism etc). I can share a paper done by the FED. They have a dictionary of positive and negative words. (Attached). I will need to know what exact words to try from you (at some point, not in a super rush).
5.	Replicate the topic modeling part of the write up. 

