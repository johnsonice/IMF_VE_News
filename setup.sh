## install some packages 
#conda install -c conda-forge spacy langdetect
#conda install gensim nltk numpy pandas jupyter notebook spyder glob2 scikit-learn seaborn lxml beautifulsoup4
## install language module 
#python -m spacy download en_core_web_lg

##
## if spacy download function doesn't work, you will have to do it the hard way
##
## pip install .tar.gz archive from path or URL
#pip install /Users/you/en_core_web_sm-2.0.0.tar.gz
## set up shortcut link to load installed package as "en_default"
#python -m spacy link en_core_web_lg en_core_web_lg


mkdir data models 
cd data 
mkdir bow_tfidf_docs doc_meta eval frequency processed_json search_terms test_dir train_dir 
mkdir eval/experts eval/word_groups search_terms/experts
cd ..
mkdir models/ngrams models/vsms


