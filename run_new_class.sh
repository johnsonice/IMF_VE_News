cd src_ft
python 01_corpus_preprocessing.py
python 02_0_corpus_doc_details.py 
python 02_1_meta_summary.py 
python 03_corpus_phrases.py 
python 04_1_vectorize_words.py 
python 06_0_frequency_country_specific_freqs.py
python 06_1_export_keywords_timeseries.py

#python corpus_tfidf.py
#python frequency_country_specific_freqs.py

#python frequency_eval.py
#python frequency_eval_expert-terms.py


## run evaluation on specific word groups 
python 07_02_frequency_eval_aggregate.py -gsf final_topic_words_small_new_class.csv -tn 10
python 07_02_frequency_eval_aggregate.py -gsf experts_refined_new_class.csv -tn 15
