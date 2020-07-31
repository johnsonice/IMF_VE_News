# Trains 6 LDA topic models
cd src_ft
python 08_02_topic_lda_train.py -top 80 # 80 Topics
python 08_02_topic_lda_train.py -top 120
python 08_02_topic_lda_train.py -top 140
python 08_02_topic_lda_train.py -top 160
python 08_02_topic_lda_train.py -top 180
python 08_02_topic_lda_train.py -top 200 # 200 topics

#python corpus_tfidf.py
#python frequency_country_specific_freqs.py

#python frequency_eval.py
#python frequency_eval_expert-terms.py


